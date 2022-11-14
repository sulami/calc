#[derive(Debug, Default)]
pub struct State {
    stack: Vec<Num>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Num {
    Int(i128),
    Float(f64),
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

#[derive(Debug, PartialEq)]
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
        _ => todo!(),
    };
    Ok(state)
}

/// Executes a chain of instructions on state. Halts at the first
/// error and returns the state at that point as well as an error
/// message.
pub fn run(state: State, ops: &[Op]) -> Result<State, (State, &'static str)> {
    ops.iter().try_fold(state, execute)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to construct a stack
    fn make_stack<I, N>(nums: I) -> Vec<Num>
    where
        I: IntoIterator<Item = N>,
        N: Into<Num>,
    {
        nums.into_iter().map(|n| n.into()).collect()
    }

    #[test]
    fn pushing_ints_works() {
        let mut state = State::default();
        state = run(state, &[Op::Push(42.into())]).expect("Failed to run ops");
        assert_eq!(state.stack, make_stack([42]));
    }

    #[test]
    fn pushing_floats_works() {
        let mut state = State::default();
        state = run(state, &[Op::Push(42.2.into())]).expect("Failed to run ops");
        assert_eq!(state.stack, make_stack([42.2]));
    }

    #[test]
    fn drop_works() {
        let mut state = State::default();
        let ops = &[Op::Push(41.into()), Op::Push(42.into()), Op::Drop];
        state = run(state, ops).expect("Failed to run ops");
        assert_eq!(state.stack, make_stack([41]));
    }

    #[test]
    fn swap_works() {
        let mut state = State::default();
        let ops = &[
            Op::Push(41.into()),
            Op::Push(42.into()),
            Op::Push(43.into()),
            Op::Swap,
        ];
        state = run(state, ops).expect("Failed to run ops");
        assert_eq!(state.stack, make_stack([41, 43, 42]));
    }

    #[test]
    fn rotate_works() {
        let mut state = State::default();
        let ops = &[
            Op::Push(41.into()),
            Op::Push(42.into()),
            Op::Push(43.into()),
            Op::Rotate,
        ];
        state = run(state, ops).expect("Failed to run ops");
        assert_eq!(state.stack, make_stack([43, 41, 42]));
    }
}
