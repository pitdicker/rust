use io;
use sys::{ReadSysCall, WriteSysCall};

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub fn new() -> Stdin { Stdin }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        Ok(ReadSysCall::perform(0, buf))
    }
}

impl Stdout {
    pub fn new() -> Stdout { Stdout }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        WriteSysCall::perform(1, buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub fn new() -> Stderr { Stderr }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        WriteSysCall::perform(2, buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = 0;

pub fn panic_output() -> Option<impl io::Write> {
    if cfg!(feature = "wasm_syscall") {
        Some(Stderr::new())
    } else {
        None
    }
}
