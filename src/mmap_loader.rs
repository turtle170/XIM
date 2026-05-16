use crate::error::{Result, XimError};
use memmap2::MmapOptions;
use std::fs::File;
use std::path::Path;

pub struct MmapLoader {
    mmap: memmap2::Mmap,
}

impl MmapLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self { mmap })
    }

    /// Expose the memory map as a slice of i16
    /// Note: This assumes the file is correctly sized and aligned.
    pub fn as_i16_slice(&self) -> Result<&[i16]> {
        if self.mmap.len() % 2 != 0 {
            return Err(XimError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Mmap length is not a multiple of 2 (i16 size)",
            )));
        }
        
        let ptr = self.mmap.as_ptr() as *const i16;
        let len = self.mmap.len() / 2;
        
        // Safety: We verified the length is a multiple of 2, and the memory map
        // guarantees the memory is accessible. We assume the data is correctly aligned.
        // In a strict environment, we'd need to ensure alignment or use unaligned reads.
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        Ok(slice)
    }
}
