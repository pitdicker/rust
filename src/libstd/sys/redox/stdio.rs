use io;
use sys::{cvt, syscall};
use sys::fd::FileDesc;

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub fn new() -> Stdin { Stdin }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let fd = FileDesc::new(0);
        let ret = fd.read(buf);
        fd.into_raw();
        ret
    }
}

impl Stdout {
    pub fn new() -> Stdout { Stdout }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let fd = FileDesc::new(1);
        let ret = fd.write(buf);
        fd.into_raw();
        ret
    }

    fn flush(&mut self) -> io::Result<()> {
        cvt(syscall::fsync(1)).and(Ok(()))
    }
}

impl Stderr {
    pub fn new() -> Stderr { Stderr }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let fd = FileDesc::new(2);
        let ret = fd.write(buf);
        fd.into_raw();
        ret
    }

    fn flush(&mut self) -> io::Result<()> {
        cvt(syscall::fsync(2)).and(Ok(()))
    }
}

pub const STDIN_BUF_SIZE: usize = ::sys_common::io::DEFAULT_BUF_SIZE;

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}
