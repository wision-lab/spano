// Heavily based on https://github.com/tracel-ai/burn/tree/v0.13.2/examples/custom-wgpu-kernel

use std::marker::PhantomData;

use burn::backend::wgpu::{
    into_contiguous, kernel_wgsl, FloatElement, GraphicsApi, IntElement, JitBackend, Kernel,
    KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, WorkGroup, WorkgroupSize,
};
use burn_tensor::ops::FloatTensor;
use derive_new::new;

// Source the kernel written in WGSL.
kernel_wgsl!(WarpRaw, "./kernel.wgsl");

// Define our kernel type with cube information.
#[derive(new, Debug)]
struct WarpKernel<E: FloatElement> {
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
            .register("background_color", "0.0, 0.0, 0.0".to_string())
            .register("padding", "1.0".to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }
}

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn into_contiguous<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D>;

    fn warp_into_tensor3(
        mapping: FloatTensor<Self, 2>,
        input: FloatTensor<Self, 3>,
        output: &mut FloatTensor<Self, 3>,
    );
}

/// Implement our custom backend trait for the existing backend `WgpuBackend`.
impl<G: GraphicsApi, F: FloatElement, I: IntElement> Backend for JitBackend<WgpuRuntime<G, F, I>> {
    fn into_contiguous<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        into_contiguous(tensor)
    }

    fn warp_into_tensor3(
        mapping: FloatTensor<Self, 2>,
        input: FloatTensor<Self, 3>,
        output: &mut FloatTensor<Self, 3>,
    ) {
        // Validate devices
        if mapping.device != input.device {
            panic!("Both tensors should be on the same device");
        }
        input.assert_is_on_same_device(&output);

        // Define workgroup size, hardcoded for simplicity.
        let workgroup_size = WorkgroupSize { x: 16, y: 16, z: 1 };

        // Create the kernel.
        let kernel = WarpKernel::<F>::new(workgroup_size);

        // For simplicity, make sure each tensor is continuous.
        let mapping = into_contiguous(mapping);
        let input = into_contiguous(input);
        if !output.is_contiguous() && output.shape.dims.iter().all(|&d| d != 1) {
            panic!("Output tensor must be contiguous");
        }

        // Build info buffer with tensor information needed by the kernel, such as shapes and strides.
        let input_shape: Vec<u32> = input.shape.dims.iter().map(|i| *i as u32).collect();
        let output_shape: Vec<u32> = output.shape.dims.iter().map(|i| *i as u32).collect();
        let input_shape_handle = mapping.client.create(bytemuck::cast_slice(&input_shape));
        let output_shape_handle = mapping.client.create(bytemuck::cast_slice(&output_shape));

        // Declare the wgsl workgroup with the number of cubes in x, y and z.
        let [num_rows, num_cols, num_channels] = output_shape[..] else {
            unreachable!("Input has 3 dims")
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
                &input_shape_handle,
                &output_shape_handle,
            ],
        );
    }
}
