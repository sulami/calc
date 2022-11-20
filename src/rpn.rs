#[derive(Clone, Debug, Default)]
pub struct State {
    pub stack: Vec<Num>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Num {
    Int(i128),
    Float(f64),
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
            (Num::Int(a), Num::Int(b)) => Num::Int(a + b),
            (Num::Int(a), Num::Float(b)) => Num::Float(a as f64 + b),
            (Num::Float(a), Num::Int(b)) => Num::Float(a + b as f64),
            (Num::Float(a), Num::Float(b)) => Num::Float(a + b),
        }
    }
}

impl std::ops::Sub for Num {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Num::Int(a), Num::Int(b)) => Num::Int(a - b),
            (Num::Int(a), Num::Float(b)) => Num::Float(a as f64 - b),
            (Num::Float(a), Num::Int(b)) => Num::Float(a - b as f64),
            (Num::Float(a), Num::Float(b)) => Num::Float(a - b),
        }
    }
}

impl std::ops::Mul for Num {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Num::Int(a), Num::Int(b)) => Num::Int(a * b),
            (Num::Int(a), Num::Float(b)) => Num::Float(a as f64 * b),
            (Num::Float(a), Num::Int(b)) => Num::Float(a * b as f64),
            (Num::Float(a), Num::Float(b)) => Num::Float(a * b),
        }
    }
}

impl std::ops::Div for Num {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Num::Int(a), Num::Int(b)) => Num::Float(a as f64 / b as f64),
            (Num::Int(a), Num::Float(b)) => Num::Float(a as f64 / b),
            (Num::Float(a), Num::Int(b)) => Num::Float(a / b as f64),
            (Num::Float(a), Num::Float(b)) => Num::Float(a / b),
        }
    }
}

impl std::ops::Neg for Num {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Num::Int(a) => Num::Int(-a),
            Num::Float(a) => Num::Float(-a),
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
    // Artithmetic
    Add,
    Subtract,
    Divide,
    Multiply,
    Negate,
    Abs,
    Modulo,
    Remainder,
    Invert,
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

/// Executes a single operation on state, returning the new state. If
/// the operation fails, returns a tuple of the old state and an error
/// message.
pub fn execute(mut state: State, op: &Op) -> Result<State, (State, &'static str)> {
    let stack_size = state.stack.len();
    match op {
        Op::Push(n) => state.stack.push(*n),
        Op::Drop if stack_size < 1 => {
            return Err((state, "stack is empty"));
        }
        Op::Drop => {
            let _ = state.stack.pop();
        }
        Op::Swap if stack_size < 2 => {
            return Err((state, "requires at least two items on stack"));
        }
        Op::Swap => state.stack.swap(stack_size - 1, stack_size - 2),
        Op::Rotate => {
            if let Some(item) = state.stack.pop() {
                state.stack.insert(0, item);
            }
        }
        Op::Add if stack_size < 2 => {
            return Err((state, "requires at least two items on stack"));
        }
        Op::Add => {
            let a = state.stack.pop().expect("failed to pop item from stack");
            let b = state.stack.pop().expect("failed to pop item from stack");
            state.stack.push(b + a);
        }
        Op::Subtract if stack_size < 2 => {
            return Err((state, "requires at least two items on stack"));
        }
        Op::Subtract => {
            let a = state.stack.pop().expect("failed to pop item from stack");
            let b = state.stack.pop().expect("failed to pop item from stack");
            state.stack.push(b - a);
        }
        Op::Multiply if stack_size < 2 => {
            return Err((state, "requires at least two items on stack"));
        }
        Op::Multiply => {
            let a = state.stack.pop().expect("failed to pop item from stack");
            let b = state.stack.pop().expect("failed to pop item from stack");
            state.stack.push(b * a);
        }
        Op::Divide if stack_size < 2 => {
            return Err((state, "requires at least two items on stack"));
        }
        Op::Divide => {
            let a = state.stack.pop().expect("failed to pop item from stack");
            let b = state.stack.pop().expect("failed to pop item from stack");
            state.stack.push(b / a);
        }
        Op::Negate if stack_size < 1 => {
            return Err((state, "stack is empty"));
        }
        Op::Negate => {
            let a = state.stack.pop().expect("failed to pop item from stack");
            state.stack.push(-a);
        }
        _ => todo!(),
    };
    Ok(state)
}

/// Executes a chain of instructions on state. Halts at the first
/// error and returns the state at that point as well as an error
/// message.
pub fn run<'a, I>(state: State, ops: I) -> Result<State, (State, &'static str)>
where
    I: IntoIterator<Item = &'a Op>,
{
    ops.into_iter().try_fold(state, execute)
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
        let state = run(State::default(), ops).expect("Failed to run ops");
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
}
