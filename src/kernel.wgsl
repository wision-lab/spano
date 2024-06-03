@group(0)
@binding(0)
var<storage, read> mapping: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read> input_shape_handle: array<u32>;

@group(0)
@binding(4)
var<storage, read> output_shape_handle: array<u32>;

const BLOCK_SIZE = vec3({{ workgroup_size_x }}u, {{ workgroup_size_y }}u, {{ workgroup_size_z }}u);
const BACKGROUND_COLOR = vec3({{ background_color }});

fn get_src_index(row: u32, col: u32, channel: u32) -> u32 {
    // Ravel multi index for HWC 
    let ncols = input_shape_handle[1];
    let nchan = input_shape_handle[2];
    let index = row * (ncols * nchan) + col * nchan + channel;
    return index;
}

fn get_dst_index(row: u32, col: u32, channel: u32) -> u32 {
    // Ravel multi index for HWC 
    let ncols = output_shape_handle[1];
    let nchan = output_shape_handle[2];
    let index = row * (ncols * nchan) + col * nchan + channel;
    return index;
}

fn get_pix_or_bkg(x: f32, y: f32, c: u32) -> f32{
    let src_rows = input_shape_handle[0];
    let src_cols = input_shape_handle[1];

    if x < 0.0 || x >= f32(src_cols) || y < 0.0 || y >= f32(src_rows) {
        return BACKGROUND_COLOR[c];
    } else {
        let index = get_src_index(u32(y), u32(x), c);
        return input[index];
    }
}

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // Indices
    let channel = workgroup_id.z;
    let row = workgroup_id.x * BLOCK_SIZE.x + (local_idx / BLOCK_SIZE.x);
    let col = workgroup_id.y * BLOCK_SIZE.x + (local_idx % BLOCK_SIZE.x);

    // Basic information
    let src_rows = input_shape_handle[0];
    let src_cols = input_shape_handle[1];
    let src_channels = input_shape_handle[2];
    let dst_rows = output_shape_handle[0];
    let dst_cols = output_shape_handle[1];
    let dst_channels = output_shape_handle[2];

    // Returns if outside the output dimension
    // This is needed for when the image size isn't a perfect multiple of workgroup size
    if row >= dst_rows || col >= dst_cols || channel >= dst_channels{
        return;
    }

    // Warp point using homogeneous coordinates
    // This would be better as a mat3x3 but I can't convert it easily
    let x_ = mapping[0]*f32(col)+mapping[1]*f32(row)+mapping[2];
    let y_ = mapping[3]*f32(col)+mapping[4]*f32(row)+mapping[5];
    let v = mapping[6]*f32(col)+mapping[7]*f32(row)+mapping[8]; 
    let x = x_/v;
    let y = y_/v;

    // If warped point isn't in input image, early exit
    let in_range_x = -{{ padding }}f <= x && x <= f32(src_cols) - 1.0 + {{ padding }}f;
    let in_range_y = -{{ padding }}f <= y && y <= f32(src_rows) - 1.0 + {{ padding }}f;

    if !in_range_x || !in_range_y {
        let index = get_dst_index(row, col, channel);
        output[index] = BACKGROUND_COLOR[channel];
        return;
    }

    // Actually do bilinear interpolation
    let left = floor(x);
    let right = left + 1.0;
    let top = floor(y);
    let bottom = top + 1.0;
    let right_weight = x - left;
    let left_weight = 1.0 - right_weight;
    let bottom_weight = y - top;
    let top_weight = 1.0 - bottom_weight;

    // Sample neighbors...
    let tl = get_pix_or_bkg(left, top, channel);
    let tr = get_pix_or_bkg(right, top, channel);
    let bl = get_pix_or_bkg(left, bottom, channel);
    let br = get_pix_or_bkg(right, bottom, channel);
    let val = clamp(top_weight * left_weight * tl + top_weight * right_weight * tr + bottom_weight * left_weight * bl + bottom_weight * right_weight * br, 0.0, 1.0);

    let index = get_dst_index(row, col, channel);
    output[index] = val;
}
