use std::collections::VecDeque;
use std::io;

use anyhow::Result;
use crossterm::{
    event::{read, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use tui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};

mod rpn;

#[derive(Default)]
struct State {
    calc_state: rpn::State,
    history: VecDeque<(rpn::Op, rpn::State)>,
    input: String,
}

fn main() -> Result<()> {
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    enable_raw_mode()?;

    event_loop(&mut terminal)?;

    execute!(stdout, LeaveAlternateScreen, DisableMouseCapture)?;
    disable_raw_mode()?;

    Ok(())
}

/// Main event loop, call once.
fn event_loop(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut state = State::default();
    terminal.draw(|f| draw_ui(&state, f))?;

    loop {
        match read()? {
            Event::Key(KeyEvent {
                code: KeyCode::Char(c),
                ..
            }) => match c {
                '0'..='9' => {
                    state.input.push(c);
                }
                'k' => state = try_op(state, rpn::Op::Drop),
                'r' => state = try_op(state, rpn::Op::Rotate),
                's' => state = try_op(state, rpn::Op::Swap),
                '+' => {
                    state = insert_input(state);
                    state = try_op(state, rpn::Op::Add);
                }
                '-' => {
                    state = insert_input(state);
                    state = try_op(state, rpn::Op::Subtract);
                }
                '*' => {
                    state = insert_input(state);
                    state = try_op(state, rpn::Op::Multiply);
                }
                '/' => {
                    state = insert_input(state);
                    state = try_op(state, rpn::Op::Divide);
                }
                '~' => state = try_op(state, rpn::Op::Negate),
                'Q' => return Ok(()),
                _ => (),
            },
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) => state = insert_input(state),
            _ => (),
        }
        terminal.draw(|f| draw_ui(&state, f))?;
    }
}

/// Updates state by inserting the current input, noop otherwise.
/// Updates history appropriately as well.
fn insert_input(mut state: State) -> State {
    if !state.input.is_empty() {
        let num = i128::from_str_radix(&state.input, 10).expect("Invalid number");
        let op = rpn::Op::Push(num.into());
        state = try_op(state, op);
        state.input.clear();
    }
    state
}

/// Attempts to apply op to the current state, returning a potentially
/// updated state. Updates history.
fn try_op(mut state: State, op: rpn::Op) -> State {
    let current_state = state.calc_state.clone();
    if let Ok(new_state) = rpn::run(state.calc_state, &[op]) {
        state.history.push_back((op, current_state));
        state.calc_state = new_state;
    } else {
        state.calc_state = current_state;
    };
    state
}

/// Redraws the UI based on the current state.
fn draw_ui(state: &State, f: &mut Frame<CrosstermBackend<io::Stdout>>) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Min(3), Constraint::Length(3)].as_ref())
        .split(f.size());

    let top_section = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(
            [
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(50),
            ]
            .as_ref(),
        )
        .split(root[0]);

    let stack_box = List::new(
        state
            .calc_state
            .stack
            .iter()
            .rev()
            .map(|i| ListItem::new(format!("{i}")))
            .collect::<Vec<ListItem>>(),
    )
    .start_corner(tui::layout::Corner::BottomLeft)
    .block(Block::default().title("Stack").borders(Borders::ALL));
    f.render_widget(stack_box, top_section[0]);

    let register_box = List::new([ListItem::new("Coming soon")])
        .block(Block::default().title("Registers").borders(Borders::ALL));
    f.render_widget(register_box, top_section[1]);

    let history_box = List::new(
        state
            .history
            .iter()
            .rev()
            .map(|(op, s)| ListItem::new(format_history_event(s, op)))
            .collect::<Vec<ListItem>>(),
    )
    .start_corner(tui::layout::Corner::BottomLeft)
    .block(Block::default().title("History").borders(Borders::ALL));
    f.render_widget(history_box, top_section[2]);

    let input_box = Paragraph::new(state.input.clone())
        .block(Block::default().title("Input").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    f.render_widget(input_box, root[1]);
}

/// Formats a history event for display to the user.
fn format_history_event(state: &rpn::State, op: &rpn::Op) -> String {
    let stack_size = state.stack.len();
    match op {
        rpn::Op::Push(n) => format!("-> {n}"),
        rpn::Op::Rotate => ">>>".to_string(),
        rpn::Op::Swap => format!(
            "{} <-> {}",
            state.stack.get(stack_size - 2).unwrap(),
            state.stack.get(stack_size - 1).unwrap()
        ),
        rpn::Op::Drop => format!("<- {}", state.stack.first().unwrap()),
        rpn::Op::Negate => format!("(-) {}", state.stack.get(stack_size - 1).unwrap()),
        rpn::Op::Add => format!(
            "{} + {}",
            state.stack.get(stack_size - 2).unwrap(),
            state.stack.get(stack_size - 1).unwrap()
        ),
        rpn::Op::Subtract => format!(
            "{} - {}",
            state.stack.get(stack_size - 2).unwrap(),
            state.stack.get(stack_size - 1).unwrap()
        ),
        rpn::Op::Multiply => format!(
            "{} × {}",
            state.stack.get(stack_size - 2).unwrap(),
            state.stack.get(stack_size - 1).unwrap()
        ),
        rpn::Op::Divide => format!(
            "{} / {}",
            state.stack.get(stack_size - 2).unwrap(),
            state.stack.get(stack_size - 1).unwrap()
        ),
        _ => format!("{state:?} -> {op:?}"),
    }
}
