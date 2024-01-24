use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{anyhow, Result};
use memmap2::Mmap;
use ndarray::{Array, Array3, ArrayView3};
use ndarray_npy::{ViewNpyError, ViewNpyExt};

use crate::utils::sorted_glob;

#[allow(dead_code)]
#[derive(Debug)]
pub struct PhotonCube<'a> {
    path: &'a str,
    bit_depth: u32,
    _storage: PhotonCubeStorage,
}

// These are needed to keep the underlying object's data in scope
// otherwise we get a use-after-free error.
// We use an enum here as either an array OR a memap object is needed.
#[derive(Debug)]
enum PhotonCubeStorage {
    ArrayStorage(Array3<u8>),
    MmapStorage(Mmap),
}

impl<'a> PhotonCube<'a> {
    pub fn view(&self) -> Result<ArrayView3<u8>, ViewNpyError> {
        match &self._storage {
            PhotonCubeStorage::ArrayStorage(arr) => Ok(arr.view()),
            PhotonCubeStorage::MmapStorage(mmap) => ArrayView3::<u8>::view_npy(mmap),
        }
    }

    pub fn open(path_str: &'a str) -> Result<Self> {
        let path = Path::new(path_str);

        if !path.exists() {
            // This should probably be a specific IO error?
            Err(anyhow!("File not found at {}!", path_str))
        } else if path.is_dir() {
            let (h, w) = (256, 512);
            let paths = sorted_glob(path, "**/*.bin")?;

            if paths.is_empty() {
                return Err(anyhow!("No .bin files found in {}!", path_str));
            }

            let mut buffer = Vec::new();

            for p in paths {
                let mut f = File::open(p)?;
                f.read_to_end(&mut buffer)?;
            }

            let t = buffer.len() / (h * w / 8);
            let arr = Array::from_vec(buffer)
                .into_shape((t, h, w / 8))?
                .mapv(|v| v.reverse_bits());

            Ok(Self {
                path: path_str,
                bit_depth: 1,
                _storage: PhotonCubeStorage::ArrayStorage(arr),
            })
        } else {
            let ext = path.extension().unwrap().to_ascii_lowercase();

            if ext != "npy" {
                // This should probably be a specific IO error?
                return Err(anyhow!(
                    "Expexted numpy array with extension `npy`, got {:?}.",
                    ext
                ));
            }

            let file = File::open(path_str)?;
            let mmap = unsafe { Mmap::map(&file)? };

            Ok(Self {
                path: path_str,
                bit_depth: 1,
                _storage: PhotonCubeStorage::MmapStorage(mmap),
            })
        }
    }
}
