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
//! state = execute(state, Op::Push(42.into()))?;
//! ```

#[derive(Clone, Debug, Default)]
pub struct State {
    pub stack: Vec<Num>,
}

impl State {
    /// Executes a single operation on state, returning the new state. If
    /// the operation fails, returns a tuple of the old state and an error
    /// message.
    pub fn execute(mut self, op: &Op) -> Result<Self, (Self, &'static str)> {
        let stack_size = self.stack.len();
        match op {
            Op::Push(n) => self.stack.push(*n),
            Op::Drop if stack_size < 1 => {
                return Err((self, "stack is empty"));
            }
            Op::Drop => {
                let _ = self.stack.pop();
            }
            Op::Swap if stack_size < 2 => {
                return Err((self, "requires at least two items on stack"));
            }
            Op::Swap => self.stack.swap(stack_size - 1, stack_size - 2),
            Op::Rotate => {
                if let Some(item) = self.stack.pop() {
                    self.stack.insert(0, item);
                }
            }
            Op::Add if stack_size < 2 => {
                return Err((self, "requires at least two items on stack"));
            }
            Op::Add => {
                let a = self.stack.pop().expect("failed to pop item from stack");
                let b = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(b + a);
            }
            Op::Subtract if stack_size < 2 => {
                return Err((self, "requires at least two items on stack"));
            }
            Op::Subtract => {
                let a = self.stack.pop().expect("failed to pop item from stack");
                let b = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(b - a);
            }
            Op::Multiply if stack_size < 2 => {
                return Err((self, "requires at least two items on stack"));
            }
            Op::Multiply => {
                let a = self.stack.pop().expect("failed to pop item from stack");
                let b = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(b * a);
            }
            Op::Divide if stack_size < 2 => {
                return Err((self, "requires at least two items on stack"));
            }
            Op::Divide => {
                let a = self.stack.pop().expect("failed to pop item from stack");
                let b = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(b / a);
            }
            Op::Negate if stack_size < 1 => {
                return Err((self, "stack is empty"));
            }
            Op::Negate => {
                let a = self.stack.pop().expect("failed to pop item from stack");
                self.stack.push(-a);
            }
            Op::Pow if stack_size < 2 => {
                return Err((self, "requires at least two items on stack"));
            }
            Op::Pow => {
                let a = self.stack.pop().expect("failed to pop item from stack");
                let b = self.stack.pop().expect("failed to pop item from stack");
                match b.pow(a) {
                    Ok(result) => self.stack.push(result),
                    Err(msg) => {
                        // Undo, push back onto stack.
                        self.stack.push(b);
                        self.stack.push(a);
                        return Err((self, msg));
                    }
                }
            }
            _ => todo!(),
        };
        Ok(self)
    }

    /// Executes a chain of instructions on state. Halts at the first
    /// error and returns the state at that point as well as an error
    /// message.
    pub fn run<'a, I>(self, ops: I) -> Result<Self, (Self, &'static str)>
    where
        I: IntoIterator<Item = &'a Op>,
    {
        ops.into_iter().try_fold(self, Self::execute)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Num {
    Int(i128),
    Float(f64),
}

impl Num {
    /// Unwrapping power, works for both i128 and f64, as well as
    /// mixes of both. Refuses negative exponents.
    fn pow(self, rhs: Self) -> Result<Self, &'static str> {
        match (self, rhs) {
            (_, Self::Int(x)) if x < 0 => Err("negative exponent"),
            (_, Self::Float(x)) if x < 0.0 => Err("negative exponent"),
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(a.pow(b as u32))),
            (Self::Float(a), Self::Int(b)) => Ok(Self::Float(a.powf(b as f64))),
            (Self::Int(a), Self::Float(b)) => Ok(Self::Float((a as f64).powf(b))),
            (Self::Float(a), Self::Float(b)) => Ok(Self::Float(a.powf(b))),
        }
    }
}

impl std::fmt::Display for Num {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
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
        let state = State::default().run(ops).expect("Failed to run ops");
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
    #[should_panic(expected = "stack is empty")]
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
    #[should_panic(expected = "requires at least two items on stack")]
    fn swap_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Swap], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least two items on stack")]
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
    #[should_panic(expected = "requires at least two items on stack")]
    fn add_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Add], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least two items on stack")]
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
    #[should_panic(expected = "requires at least two items on stack")]
    fn subtract_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Subtract], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least two items on stack")]
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
    #[should_panic(expected = "requires at least two items on stack")]
    fn multiply_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Multiply], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least two items on stack")]
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
    #[should_panic(expected = "requires at least two items on stack")]
    fn divide_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Divide], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least two items on stack")]
    fn divide_errors_with_stack_of_one() {
        run_and_compare_stack(&[Op::Push(42.into()), Op::Divide], [42]);
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
    #[should_panic(expected = "stack is empty")]
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
    #[should_panic(expected = "requires at least two items on stack")]
    fn pow_errors_on_empty_stack() {
        run_and_compare_stack(&[Op::Pow], [42]);
    }

    #[test]
    #[should_panic(expected = "requires at least two items on stack")]
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
        match State::default().run(ops) {
            Err((state, _)) => assert_eq!(state.stack, make_stack([2, -4])),
            _ => panic!("ops ran successfully"),
        }
    }
}
