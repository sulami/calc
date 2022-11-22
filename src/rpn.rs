//! A reverse Polish notiation calculator core.
//!
//! This module is meant to be useful in isolation, and should be
//! usable to build different calculator frontends or otherwise
//! include RPN calculator functionality in existing programs.
//!
//! Example usage:
//!
//! ```
//! let mut state = State::default();
//! state = state.execute(Op::Push(42.into()))?;
//! ```

#![allow(dead_code)]

use std::fmt::{self, Debug, Display};

#[derive(Clone, Debug, Default)]
pub struct State {
    stack: Vec<Num>,
}

impl State {
    /// Returns an error if the stack size is less than required.
    fn require_stack(&self, size: usize) -> Result<(), RPNError> {
        if self.stack.len() >= size {
            Ok(())
        } else {
            Err(RPNError::RequiresStack(size))
        }
    }

    /// Returns a (cloned) Vec of the internal stack, from least
    /// recently to most recently pushed.
    pub fn stack_vec(&self) -> Vec<Num> {
        self.stack.clone()
    }

    /// Returns the length of the internal stack.
    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }

    /// Attempt to fetch an item from the internal stack. Note that
    /// the most recently pushed item is at the end.
    pub fn stack_get(&self, idx: usize) -> Option<&Num> {
        self.stack.get(idx)
    }

    /// Attempt to fetch the last (most recently pushed) item from the
    /// internal stack.
    pub fn stack_last(&self) -> Option<&Num> {
        self.stack.last()
    }

    /// Executes a single operation on state, returning the new state. If
    /// the operation fails, returns a tuple of the old state and an error
    /// message.
    pub fn execute(&mut self, op: &Op) -> Result<(), RPNError> {
        let stack_size = self.stack.len();

        match op {
            Op::Push(n) => self.stack.push(*n),
            Op::Drop => {
                self.require_stack(1)?;
                let _ = self.stack.pop();
            }
            Op::Swap => {
                self.require_stack(2)?;
                self.stack.swap(stack_size - 1, stack_size - 2)
            }
            Op::Rotate => {
                if let Some(item) = self.stack.pop() {
                    self.stack.insert(0, item);
                }
            }
            Op::Clear => self.stack.clear(),
            Op::Add => {
                self.require_stack(2)?;
                let b = self.stack.pop().expect("failed to pop item from stack");
                let a = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(a + b);
            }
            Op::Subtract => {
                self.require_stack(2)?;
                let b = self.stack.pop().expect("failed to pop item from stack");
                let a = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(a - b);
            }
            Op::Multiply => {
                self.require_stack(2)?;
                let b = self.stack.pop().expect("failed to pop item from stack");
                let a = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(a * b);
            }
            Op::Divide => {
                self.require_stack(2)?;
                if self.stack_last().unwrap().is_zero() {
                    return Err(RPNError::DivisionByZero);
                }
                let b = self.stack.pop().expect("failed to pop item from stack");
                let a = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(a / b);
            }
            Op::Negate => {
                self.require_stack(1)?;
                let a = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(-a);
            }
            Op::Pow => {
                self.require_stack(2)?;
                let a = self
                    .stack
                    .get(stack_size - 2)
                    .expect("failed to peek at stack");
                let b = self
                    .stack
                    .get(stack_size - 1)
                    .expect("failed to peek at stack");
                let result = a.pow(*b)?;
                let _ = self.stack.pop();
                let _ = self.stack.pop();
                self.stack.push(result);
            }
            Op::Absolute => {
                self.require_stack(1)?;
                let a = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(a.abs());
            }
            Op::Modulo => {
                self.require_stack(2)?;
                let a = self
                    .stack
                    .get(stack_size - 2)
                    .expect("failed to peek at stack");
                let b = self
                    .stack
                    .get(stack_size - 1)
                    .expect("failed to peek at stack");
                let result = a.modulo(*b)?;
                let _ = self.stack.pop();
                let _ = self.stack.pop();
                self.stack.push(result);
            }
            Op::Remainder => {
                self.require_stack(2)?;
                let a = self
                    .stack
                    .get(stack_size - 2)
                    .expect("failed to peek at stack");
                let b = self
                    .stack
                    .get(stack_size - 1)
                    .expect("failed to peek at stack");
                let result = a.rem(*b)?;
                let _ = self.stack.pop();
                let _ = self.stack.pop();
                self.stack.push(result);
            }
            _ => todo!(),
        };
        Ok(())
    }

    /// Executes a chain of instructions on state. Halts at the first
    /// error and returns the state at that point as well as an error
    /// message.
    pub fn run<'a, I>(&mut self, ops: I) -> Result<(), RPNError>
    where
        I: IntoIterator<Item = &'a Op>,
    {
        // ops.into_iter().try_fold(self, Self::execute)
        for op in ops {
            self.execute(op)?
        }
        Ok(())
    }
}

/// The errors that can occur when trying to process an Op.
pub enum RPNError {
    /// An operation required a certain stack size, which was not met.
    RequiresStack(usize),
    /// Any of the dividing operations received a zero divisor.
    DivisionByZero,
    /// Pow received a negative exponent.
    NegativePow,
}

impl std::error::Error for RPNError {}

impl Debug for RPNError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::RequiresStack(size) if *size == 1 => {
                write!(f, "requires at least {size} item on the stack")
            }
            Self::RequiresStack(size) => write!(f, "requires at least {size} items on the stack"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::NegativePow => write!(f, "negative exponent"),
        }
    }
}

impl Display for RPNError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Num {
    Int(i128),
    Float(f64),
}

impl Num {
    /// True if the wrapped value is zero.
    fn is_zero(&self) -> bool {
        match self {
            Self::Int(0) => true,
            Self::Float(f) => *f == 0.0,
            _ => false,
        }
    }

    /// Unwrapping power, works for both i128 and f64, as well as
    /// mixes of both. Refuses negative exponents.
    fn pow(self, rhs: Self) -> Result<Self, RPNError> {
        match (self, rhs) {
            (_, Self::Int(x)) if x < 0 => Err(RPNError::NegativePow),
            (_, Self::Float(x)) if x < 0.0 => Err(RPNError::NegativePow),
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(a.pow(b as u32))),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(a.powf(b as f64))),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float((a as f64).powf(b))),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(a.powf(b))),
        }
    }

    /// Returns the absolute value.
    fn abs(self) -> Self {
        match self {
            Self::Int(n) => Self::Int(n.abs()),
            Self::Float(n) => Self::Float(n.abs()),
        }
    }

    /// Returns the modulo.
    fn modulo(self, rhs: Self) -> Result<Self, RPNError> {
        match (self, rhs) {
            (_, Self::Int(0)) => Err(RPNError::DivisionByZero),
            (_, Self::Float(b)) if b == 0.0 => Err(RPNError::DivisionByZero),
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(a.div_euclid(b))),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float((a as f64).div_euclid(b))),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(a.div_euclid(b as f64))),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(a.div_euclid(b))),
        }
    }

    /// Returns the remainder.
    fn rem(self, rhs: Self) -> Result<Self, RPNError> {
        match (self, rhs) {
            (_, Self::Int(0)) => Err(RPNError::DivisionByZero),
            (_, Self::Float(b)) if b == 0.0 => Err(RPNError::DivisionByZero),
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(a.rem_euclid(b))),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float((a as f64).rem_euclid(b))),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(a.rem_euclid(b as f64))),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(a.rem_euclid(b))),
        }
    }
}

impl Display for Num {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Int(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
        }
    }
}

impl From<i128> for Num {
    fn from(source: i128) -> Self {
        Self::Int(source)
    }
}

impl From<f64> for Num {
    fn from(source: f64) -> Self {
        Self::Float(source)
    }
}

impl std::ops::Add for Num {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Self::Int(a + b),
            (Self::Int(a), Self::Float(b)) => Self::Float(a as f64 + b),
            (Self::Float(a), Self::Int(b)) => Self::Float(a + b as f64),
            (Self::Float(a), Self::Float(b)) => Self::Float(a + b),
        }
    }
}

impl std::ops::Sub for Num {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Self::Int(a - b),
            (Self::Int(a), Self::Float(b)) => Self::Float(a as f64 - b),
            (Self::Float(a), Self::Int(b)) => Self::Float(a - b as f64),
            (Self::Float(a), Self::Float(b)) => Self::Float(a - b),
        }
    }
}

impl std::ops::Mul for Num {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Self::Int(a * b),
            (Self::Int(a), Self::Float(b)) => Self::Float(a as f64 * b),
            (Self::Float(a), Self::Int(b)) => Self::Float(a * b as f64),
            (Self::Float(a), Self::Float(b)) => Self::Float(a * b),
        }
    }
}

impl std::ops::Div for Num {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Int(a), Self::Int(b)) => Self::Float(a as f64 / b as f64),
            (Self::Int(a), Self::Float(b)) => Self::Float(a as f64 / b),
            (Self::Float(a), Self::Int(b)) => Self::Float(a / b as f64),
            (Self::Float(a), Self::Float(b)) => Self::Float(a / b),
        }
    }
}

impl std::ops::Neg for Num {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Int(a) => Self::Int(-a),
            Self::Float(a) => Self::Float(-a),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Op {
    // Stack manipulation
    Push(Num),
    Drop,
    Swap,
    Rotate,
    Clear,
    // Artithmetic
    Add,
    Subtract,
    Divide,
    Multiply,
    Negate,
    Absolute,
    Modulo,
    Remainder,
    Invert,
    Pow,
    SquareRoot,
    Sine,
    Cosine,
    Tangent,
    // Bitwise operations
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNand,
    BitwiseNOT,
    ShiftLeft,
    ShiftRight,
    // Misc
    Rand,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to construct a stack.
    fn make_stack<I, N>(nums: I) -> Vec<Num>
    where
        I: IntoIterator<Item = N>,
        N: Into<Num>,
    {
        nums.into_iter().map(Into::into).collect()
    }

    /// Helper function to run a set of operations, then compare the
    /// stack against an expected one. Will propagate errors.
    fn run_and_compare_stack<'a, I, J, N>(ops: I, expected: J)
    where
        I: IntoIterator<Item = &'a Op>,
        J: IntoIterator<Item = N>,
        N: Into<Num>,
    {
        let mut state = State::default();
        state.run(ops).expect("Failed to run ops");
        assert_eq!(state.stack, make_stack(expected));
    }

    #[test]
    fn push_for_ints_works() {
        run_and_compare_stack(&[Op::Push(42.into())], [42]);
    }

    #[test]
    fn push_for_floats_works() {
        run_and_compare_stack(&[Op::Push(42.2.into())], [42.2]);
    }

    #[test]
    fn drop_works() {
        run_and_compare_stack(&[Op::Push(41.into()), Op::Push(42.into()), Op::Drop], [41]);
    }

    #[test]
    #[should_panic(expected = "requires at least 1 item on the stack")]
    fn drop_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Drop], [42]);
    }

    #[test]
    fn swap_works() {
        run_and_compare_stack(
            &[
                Op::Push(41.into()),
                Op::Push(42.into()),
                Op::Push(43.into()),
                Op::Swap,
            ],
            [41, 43, 42],
        );
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn swap_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Swap], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn swap_errors_with_stack_of_one() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Swap], [42]);
    }

    #[test]
    fn rotate_works() {
        run_and_compare_stack(
            &[
                Op::Push(41.into()),
                Op::Push(42.into()),
                Op::Push(43.into()),
                Op::Rotate,
            ],
            [43, 41, 42],
        );
    }

    #[test]
    fn rotate_is_noop_on_empty_stack() {
        run_and_compare_stack(&[Op::Rotate], Vec::<Num>::new());
    }

    #[test]
    fn rotate_is_noop_on_stack_of_one() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Rotate], [42]);
    }

    #[test]
    fn clear_clears_stack() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Clear], Vec::<Num>::new());
    }

    #[test]
    fn clear_noop_on_empty_stack() {
        run_and_compare_stack(&[Op::Clear], Vec::<Num>::new());
    }

    #[test]
    fn add_works() {
        run_and_compare_stack(&[Op::Push(20.into()), Op::Push(22.into()), Op::Add], [42]);
    }

    #[test]
    fn add_works_for_mixed_types() {
        run_and_compare_stack(
            &[Op::Push(41.into()), Op::Push(1.5.into()), Op::Add],
            [42.5],
        );
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn add_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Add], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn add_errors_with_stack_of_one() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Add], [42]);
    }

    #[test]
    fn subtract_works() {
        run_and_compare_stack(
            &[Op::Push(62.into()), Op::Push(20.into()), Op::Subtract],
            [42],
        );
    }

    #[test]
    fn subtract_works_for_mixed_types() {
        run_and_compare_stack(
            &[Op::Push(44.into()), Op::Push(1.5.into()), Op::Subtract],
            [42.5],
        );
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn subtract_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Subtract], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn subtract_errors_with_stack_of_one() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Subtract], [42]);
    }

    #[test]
    fn multiply_works() {
        run_and_compare_stack(
            &[Op::Push(21.into()), Op::Push(2.into()), Op::Multiply],
            [42],
        );
    }

    #[test]
    fn multiply_works_for_mixed_types() {
        run_and_compare_stack(
            &[Op::Push(3.into()), Op::Push(1.5.into()), Op::Multiply],
            [4.5],
        );
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn multiply_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Multiply], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn multiply_errors_with_stack_of_one() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Multiply], [42]);
    }

    #[test]
    fn divide_works() {
        run_and_compare_stack(
            &[Op::Push(12.into()), Op::Push(4.into()), Op::Divide],
            [3.0],
        );
    }

    #[test]
    fn divide_works_for_mixed_types() {
        run_and_compare_stack(
            &[Op::Push(3.into()), Op::Push(1.5.into()), Op::Divide],
            [2.0],
        );
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn divide_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Divide], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn divide_errors_with_stack_of_one() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Divide], [42]);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn divide_errors_zero_divisor() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Push(0.into()), Op::Divide], [42]);
    }

    #[test]
    fn negate_for_ints_works() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Negate], [-42]);
    }

    #[test]
    fn negate_for_floats_works() {
        run_and_compare_stack(&[Op::Push(42.2.into()), Op::Negate], [-42.2]);
    }

    #[test]
    #[should_panic(expected = "requires at least 1 item on the stack")]
    fn negate_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Negate], [42]);
    }

    #[test]
    fn pow_works() {
        run_and_compare_stack(&[Op::Push(2.into()), Op::Push(4.into()), Op::Pow], [16]);
    }

    #[test]
    fn pow_works_with_float_exponent() {
        run_and_compare_stack(&[Op::Push(2.into()), Op::Push(4.0.into()), Op::Pow], [16.0]);
    }

    #[test]
    fn pow_works_with_float_base() {
        run_and_compare_stack(&[Op::Push(2.0.into()), Op::Push(4.into()), Op::Pow], [16.0]);
    }

    #[test]
    fn pow_works_with_float_both() {
        run_and_compare_stack(
            &[Op::Push(2.0.into()), Op::Push(4.0.into()), Op::Pow],
            [16.0],
        );
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn pow_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Pow], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn pow_errors_with_stack_of_one() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Pow], [42]);
    }

    #[test]
    #[should_panic(expected = "negative exponent")]
    fn pow_refuses_negative_exponent() {
        run_and_compare_stack(
            &[Op::Push(2.into()), Op::Push((-4).into()), Op::Pow],
            [16.0],
        );
    }

    #[test]
    #[should_panic(expected = "negative exponent")]
    fn pow_refuses_negative_float_exponent() {
        run_and_compare_stack(
            &[Op::Push(2.into()), Op::Push((-4.0).into()), Op::Pow],
            [16.0],
        );
    }

    #[test]
    fn pow_keeps_stack_intact_on_error() {
        let ops = &[Op::Push(2.into()), Op::Push((-4).into()), Op::Pow];
        let mut state = State::default();
        match state.run(ops) {
            Err(_) => assert_eq!(state.stack, make_stack([2, -4])),
            _ => panic!("ops ran successfully"),
        }
    }

    #[test]
    fn absolute_works() {
        run_and_compare_stack(&[Op::Push((-42).into()), Op::Absolute], [42]);
    }

    #[test]
    fn absolute_works_for_floats() {
        run_and_compare_stack(&[Op::Push((-42.5).into()), Op::Absolute], [42.5]);
    }

    #[test]
    fn mod_works() {
        run_and_compare_stack(&[Op::Push(10.into()), Op::Push(3.into()), Op::Modulo], [3]);
    }

    #[test]
    fn mod_works_with_float_divisor() {
        run_and_compare_stack(
            &[Op::Push(10.into()), Op::Push(3.0.into()), Op::Modulo],
            [3.0],
        );
    }

    #[test]
    fn mod_works_with_float_dividend() {
        run_and_compare_stack(
            &[Op::Push(10.0.into()), Op::Push(3.into()), Op::Modulo],
            [3.0],
        );
    }

    #[test]
    fn mod_works_with_float_both() {
        run_and_compare_stack(
            &[Op::Push(10.0.into()), Op::Push(3.0.into()), Op::Modulo],
            [3.0],
        );
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn mod_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Modulo], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn mod_errors_with_stack_of_one() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Modulo], [42]);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn mod_refuses_zero_divisor() {
        run_and_compare_stack(
            &[Op::Push(2.into()), Op::Push((0).into()), Op::Modulo],
            [16.0],
        );
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn mod_refuses_negative_float_divisor() {
        run_and_compare_stack(
            &[Op::Push(2.into()), Op::Push((0.0).into()), Op::Modulo],
            [16.0],
        );
    }

    #[test]
    fn mod_keeps_stack_intact_on_error() {
        let ops = &[Op::Push(2.into()), Op::Push((0).into()), Op::Modulo];
        let mut state = State::default();
        match state.run(ops) {
            Err(_) => assert_eq!(state.stack, make_stack([2, 0])),
            _ => panic!("ops ran successfully"),
        }
    }

    #[test]
    fn rem_works() {
        run_and_compare_stack(
            &[Op::Push(10.into()), Op::Push(3.into()), Op::Remainder],
            [1],
        );
    }

    #[test]
    fn rem_works_with_float_divisor() {
        run_and_compare_stack(
            &[Op::Push(10.into()), Op::Push(3.0.into()), Op::Remainder],
            [1.0],
        );
    }

    #[test]
    fn rem_works_with_float_dividend() {
        run_and_compare_stack(
            &[Op::Push(10.0.into()), Op::Push(3.into()), Op::Remainder],
            [1.0],
        );
    }

    #[test]
    fn rem_works_with_float_both() {
        run_and_compare_stack(
            &[Op::Push(10.0.into()), Op::Push(3.0.into()), Op::Remainder],
            [1.0],
        );
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn rem_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Remainder], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least 2 items on the stack")]
    fn rem_errors_with_stack_of_one() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Remainder], [42]);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn rem_refuses_zero_divisor() {
        run_and_compare_stack(
            &[Op::Push(2.into()), Op::Push((0).into()), Op::Remainder],
            [16.0],
        );
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn rem_refuses_negative_float_divisor() {
        run_and_compare_stack(
            &[Op::Push(2.into()), Op::Push((0.0).into()), Op::Remainder],
            [16.0],
        );
    }

    #[test]
    fn rem_keeps_stack_intact_on_error() {
        let ops = &[Op::Push(2.into()), Op::Push((0).into()), Op::Remainder];
        let mut state = State::default();
        match state.run(ops) {
            Err(_) => assert_eq!(state.stack, make_stack([2, 0])),
            _ => panic!("ops ran successfully"),
        }
    }
}
