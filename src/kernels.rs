// Heavily based on https://github.com/tracel-ai/burn/tree/v0.13.2/examples/custom-wgpu-kernel

use std::marker::PhantomData;

use burn::backend::wgpu::{
    into_contiguous, kernel_wgsl, FloatElement, GraphicsApi, IntElement, JitBackend, Kernel,
    KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, WorkGroup, WorkgroupSize,
};
use burn_tensor::{
    ops::{BoolTensor, FloatTensor, FloatTensorOps},
    Shape, Tensor,
};
use derive_new::new;

use crate::warps::Mapping;

// Source the kernel written in WGSL.
kernel_wgsl!(WarpRaw, "./warp_kernel.wgsl");
kernel_wgsl!(BlendRaw, "./blend_kernel.wgsl");

// Define our kernel type with cube information.
#[derive(new, Debug)]
struct WarpKernel<E: FloatElement> {
    cube_dim: WorkgroupSize,
    _elem: PhantomData<E>,
}

#[derive(new, Debug)]
struct BlendKernel<E: FloatElement> {
    cube_dim: WorkgroupSize,
    _elem: PhantomData<E>,
}

// Implement the dynamic kernel trait for our kernel type.
impl<E: FloatElement> KernelSource for WarpKernel<E> {
    fn source(&self) -> SourceTemplate {
        // Extend our raw kernel with cube size information using the
        // `SourceTemplate` trait.
        WarpRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("workgroup_size_z", self.cube_dim.z.to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }
}

impl<E: FloatElement> KernelSource for BlendKernel<E> {
    fn source(&self) -> SourceTemplate {
        // Extend our raw kernel with cube size information using the
        // `SourceTemplate` trait.
        BlendRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("workgroup_size_z", self.cube_dim.z.to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }
}

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn warp_into_tensor3(
        mapping: Mapping<Self>,
        input: Tensor<Self, 3>,
        output: &mut FloatTensor<Self, 3>,
        valid: &mut BoolTensor<Self, 2>,
        background: Option<Tensor<Self, 1>>,
    );

    fn blend_into_tensor3(
        mappings: &[Mapping<Self>],
        inputs: &[Tensor<Self, 3>],
        weights: Tensor<Self, 3>,
        output: &mut FloatTensor<Self, 3>,
    );
}

/// Implement our custom backend trait for the existing backend `WgpuBackend`.
impl<G: GraphicsApi, F: FloatElement, I: IntElement> Backend for JitBackend<WgpuRuntime<G, F, I>> {
    fn warp_into_tensor3(
        mapping: Mapping<Self>,
        input: Tensor<Self, 3>,
        output: &mut FloatTensor<Self, 3>,
        valid: &mut BoolTensor<Self, 2>,
        background: Option<Tensor<Self, 1>>,
    ) {
        // Validate devices
        if mapping.device() != input.device() {
            panic!("Both tensors should be on the same device");
        }
        if output.device != input.device() {
            panic!("Both tensors should be on the same device");
        }

        // For simplicity, make sure each tensor is continuous.
        let mapping = into_contiguous(mapping.mat.into_primitive());
        let input = into_contiguous(input.into_primitive());
        if !output.is_contiguous() && output.shape.dims.iter().all(|&d| d != 1) {
            panic!("Output tensor must be contiguous");
        }
        if !valid.is_contiguous() && valid.shape.dims.iter().all(|&d| d != 1) {
            panic!("Valid tensor must be contiguous");
        }

        // Validate sizes.
        if output.shape.dims[0] != valid.shape.dims[0]
            || output.shape.dims[1] != valid.shape.dims[1]
        {
            panic!("Output and Valid tensor should have same height and width");
        }

        // Find padding and background values
        let bkg = if let Some(bkg) = background {
            Self::float_cat(
                vec![
                    bkg.into_primitive(),
                    Self::float_ones(Shape::new([1]), &output.device),
                ],
                0,
            )
        } else {
            Self::float_zeros(Shape::new([input.shape.dims[2] + 1]), &output.device)
        };

        // Define workgroup size, hardcoded for simplicity.
        let workgroup_size = WorkgroupSize { x: 16, y: 16, z: 1 };

        // Create the kernel.
        let kernel = WarpKernel::<F>::new(workgroup_size);

        // Build info buffer with tensor information needed by the kernel, such as shapes and strides.
        let input_shape: Vec<u32> = input.shape.dims.iter().map(|i| *i as u32).collect();
        let output_shape: Vec<u32> = output.shape.dims.iter().map(|i| *i as u32).collect();
        let input_shape_handle = mapping.client.create(bytemuck::cast_slice(&input_shape));
        let output_shape_handle = mapping.client.create(bytemuck::cast_slice(&output_shape));

        // Declare the wgsl workgroup with the number of cubes in x, y and z.
        let [num_rows, num_cols, num_channels] = output_shape[..] else {
            unreachable!("Input should have 3 dims")
        };
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / workgroup_size.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / workgroup_size.y as f32) as u32;
        let cubes_needed_in_z = f32::ceil(num_channels as f32 / workgroup_size.z as f32) as u32;
        let cube_count = WorkGroup::new(cubes_needed_in_x, cubes_needed_in_y, cubes_needed_in_z);

        // Execute lazily the kernel with the launch information and the given buffers.
        mapping.client.execute(
            Kernel::Custom(Box::new(SourceKernel::new(
                kernel,
                cube_count,
                workgroup_size,
            ))),
            &[
                &mapping.handle,
                &input.handle,
                &output.handle,
                &valid.handle,
                &input_shape_handle,
                &output_shape_handle,
                &bkg.handle,
            ],
        );
    }

    fn blend_into_tensor3(
        mappings: &[Mapping<Self>],
        inputs: &[Tensor<Self, 3>],
        weights: Tensor<Self, 3>,
        output: &mut FloatTensor<Self, 3>,
    ) {
        // Assume devices have already been validated
        // Assume output has one extra channel for weights

        // Assert output is contiguous
        if !output.is_contiguous() && output.shape.dims.iter().all(|&d| d != 1) {
            panic!("Output tensor must be contiguous");
        }

        // Define workgroup size, hardcoded for simplicity.
        let workgroup_size = WorkgroupSize { x: 16, y: 16, z: 1 };

        // Declare the wgsl workgroup with the number of cubes in x, y and z.
        let output_shape: Vec<u32> = output.shape.dims.iter().map(|i| *i as u32).collect();
        let output_shape_handle = output.client.create(bytemuck::cast_slice(&output_shape));
        let [num_rows, num_cols, num_channels] = output_shape[..] else {
            unreachable!("Input has 3 dims")
        };
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / workgroup_size.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / workgroup_size.y as f32) as u32;
        let cubes_needed_in_z = f32::ceil(num_channels as f32 / workgroup_size.z as f32) as u32;
        let cube_count = WorkGroup::new(cubes_needed_in_x, cubes_needed_in_y, cubes_needed_in_z);

        // Main loop where each frame is warped and blended onto buffer
        for (input, map) in inputs.iter().zip(mappings) {
            // Get linear blending weights and concatenate them to the input as last channel
            let input = Tensor::cat(vec![input.clone(), weights.clone()], 2);

            // For simplicity, make sure each tensor is continuous.
            let mapping = into_contiguous(map.clone().mat.into_primitive());
            let input = into_contiguous(input.clone().into_primitive());

            // Build info buffer with tensor information needed by the kernel, such as shapes and strides.
            let input_shape: Vec<u32> = input.shape.dims.iter().map(|i| *i as u32).collect();
            let input_shape_handle = mapping.client.create(bytemuck::cast_slice(&input_shape));

            // Execute lazily the kernel with the launch information and the given buffers.
            mapping.client.execute(
                Kernel::Custom(Box::new(SourceKernel::new(
                    BlendKernel::<F>::new(workgroup_size),
                    cube_count.clone(),
                    workgroup_size,
                ))),
                &[
                    &mapping.handle,
                    &input.handle,
                    &output.handle,
                    &input_shape_handle,
                    &output_shape_handle,
                ],
            );
        }
    }
}
