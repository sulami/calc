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
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table, Wrap},
    Frame, Terminal,
};

use rpn::Op;

mod rpn;

/// The global app state.
#[derive(Default)]
struct State {
    calc_state: rpn::State,
    history: Vec<(Op, rpn::State)>,
    mode: InputMode,
    input: String,
    message: Option<String>,
}

/// The different main modes the app can be in. Defines the
/// keybindings, and in some cases display.
enum InputMode {
    // TODO Split into different bases. Implement hex, oct, and bin
    // format for Num. Factor out input handling to accept different
    // sets of number keys.
    Normal,
    StoreRegister,
    RecallRegister,
    Help,
}

impl Default for InputMode {
    fn default() -> Self {
        Self::Normal
    }
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
    let mut state = State {
        message: Some("hit h for help".to_string()),
        ..State::default()
    };

    terminal.draw(|f| draw_ui(&state, f))?;

    loop {
        match state.mode {
            InputMode::Normal => {
                match read()? {
                    Event::Key(KeyEvent {
                        code: KeyCode::Char(c),
                        ..
                    }) => match c {
                        '0'..='9' => state.input.push(c),
                        '.' => {
                            if !state.input.contains('.') {
                                state.input.push(c)
                            }
                        }
                        'A' => state = try_op(state, Op::Absolute),
                        'C' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Cosine);
                        }
                        'D' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Floor);
                        }
                        'E' => state = try_op(state, Op::Push(std::f64::consts::E.into())),
                        'h' => state.mode = InputMode::Help,
                        'k' => state = try_op(state, Op::Drop),
                        'L' => {
                            state = insert_input(state);
                            state = try_op(state, Op::NaturalLogarithm);
                        }
                        'n' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Negate);
                        }
                        'P' => state = try_op(state, Op::Push(std::f64::consts::PI.into())),
                        'r' => state = try_op(state, Op::Rotate),
                        'R' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Remainder);
                        }
                        's' => state = try_op(state, Op::Swap),
                        'S' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Sine);
                        }
                        'T' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Tangent);
                        }
                        'u' => state = undo(state),
                        'U' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Ceiling);
                        }
                        'V' => {
                            state = insert_input(state);
                            state = try_op(state, Op::SquareRoot);
                        }
                        'x' => {
                            state = insert_input(state);
                            state.message = Some("enter register name".to_string());
                            state.mode = InputMode::StoreRegister;
                        }
                        'y' => {
                            state = insert_input(state);
                            state.message = Some("enter register name".to_string());
                            state.mode = InputMode::RecallRegister;
                        }
                        '+' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Add);
                        }
                        '-' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Subtract);
                        }
                        '*' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Multiply);
                        }
                        '/' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Divide);
                        }
                        '^' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Pow);
                        }
                        '%' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Modulo);
                        }
                        '~' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Round);
                        }
                        '\\' => {
                            state = insert_input(state);
                            state = try_op(state, Op::Invert);
                        }
                        '#' => state = try_op(state, Op::Push(rand::random::<f64>().into())),
                        'Q' => return Ok(()),
                        _ => (),
                    },
                    Event::Key(KeyEvent { code: key, .. }) => match key {
                        KeyCode::Enter => {
                            state = if state.input.is_empty() {
                                push(state)
                            } else {
                                insert_input(state)
                            }
                        }
                        KeyCode::Backspace => {
                            if !state.input.is_empty() {
                                let _ = state.input.pop();
                            }
                        }
                        KeyCode::Delete => state = try_op(state, Op::Clear),
                        _ => (),
                    },
                    _ => (),
                }

                terminal.draw(|f| draw_ui(&state, f))?;

                if state.message.is_some() {
                    state.message = None;
                }
            }
            InputMode::StoreRegister => {
                if let Event::Key(KeyEvent {
                    code: KeyCode::Char(c),
                    ..
                }) = read()?
                {
                    state.mode = InputMode::Normal;
                    state.message = None;
                    match c {
                        'a'..='z' => state = try_op(state, Op::Store(c)),
                        'A'..='Z' => state = try_op(state, Op::Store(c)),
                        _ => state.message = Some("invalid register name".to_string()),
                    }
                }
                terminal.draw(|f| draw_ui(&state, f))?;
            }
            InputMode::RecallRegister => {
                if let Event::Key(KeyEvent {
                    code: KeyCode::Char(c),
                    ..
                }) = read()?
                {
                    state.mode = InputMode::Normal;
                    state.message = None;
                    match c {
                        'a'..='z' => state = try_op(state, Op::Recall(c)),
                        'A'..='Z' => state = try_op(state, Op::Recall(c)),
                        _ => state.message = Some("invalid register name".to_string()),
                    }
                }
                terminal.draw(|f| draw_ui(&state, f))?;
            }
            InputMode::Help => {
                if let Event::Key(KeyEvent {
                    code: KeyCode::Char('h'),
                    ..
                }) = read()?
                {
                    state.mode = InputMode::Normal;
                }
                terminal.draw(|f| draw_ui(&state, f))?;
            }
        }
    }
}

/// If the stack is non-empty, push the last element again, noop
/// otherwise.
fn push(mut state: State) -> State {
    if let Some(&num) = state.calc_state.stack_last() {
        let op = Op::Push(num);
        state = try_op(state, op);
    }
    state
}

/// Inserts the current input if any, noop otherwise.
fn insert_input(mut state: State) -> State {
    if !state.input.is_empty() {
        let num = state
            .input
            .parse::<i128>()
            .map(rpn::Num::from)
            .or_else(|_| state.input.parse::<f64>().map(rpn::Num::from))
            .expect("Invalid number");
        let op = Op::Push(num);
        state = try_op(state, op);
        state.input.clear();
    }
    state
}

/// Attempts to apply op to the current state, returning a potentially
/// updated state. Updates history.
fn try_op(mut state: State, op: Op) -> State {
    let current_state = state.calc_state.clone();
    match state.calc_state.execute(&op) {
        Ok(()) => state.history.push((op, current_state)),
        Err(error) => state.message = Some(format!("{error}")),
    }
    state
}

/// If there is any history, pops off the last entry and resets to
/// that state.
fn undo(mut state: State) -> State {
    if let Some((_op, new_state)) = state.history.pop() {
        state.calc_state = new_state;
    }
    state
}

/// Redraws the UI based on the current state.
fn draw_ui(state: &State, f: &mut Frame<CrosstermBackend<io::Stdout>>) {
    match state.mode {
        InputMode::Help => draw_help_screen(f),
        _ => draw_default_screen(state, f),
    }
}

/// Draws the help screen interface.
fn draw_help_screen(f: &mut Frame<CrosstermBackend<io::Stdout>>) {
    let root = Layout::default()
        .margin(1)
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(f.size());

    let header = Row::new(vec![Cell::from("Key"), Cell::from("Action")]).bottom_margin(1);

    let left_table = Table::new(vec![
        Row::new(vec!["0-9, a-z", "number input"]),
        Row::new(vec!["enter", "push to stack"]),
        Row::new(vec!["u", "undo"]),
        Row::new(vec!["h", "toggle help screen"]),
        Row::new(vec!["Q", "quit"]).bottom_margin(1),
        Row::new(vec!["k", "drop"]),
        Row::new(vec!["s", "swap"]),
        Row::new(vec!["R", "rotate stack"]).bottom_margin(1),
        Row::new(vec!["x <key>", "store to register"]),
        Row::new(vec!["y <key>", "recall from register"]).bottom_margin(1),
        Row::new(vec!["E", "push euler's number"]),
        Row::new(vec!["P", "push pi"]),
        Row::new(vec!["#", "push random float 0<f<1"]),
    ])
    .header(header.clone())
    .column_spacing(1)
    .widths([Constraint::Length(8), Constraint::Percentage(90)].as_ref());
    f.render_widget(left_table, root[0]);

    let right_table = Table::new(vec![
        Row::new(vec!["+", "add"]),
        Row::new(vec!["-", "subtract"]),
        Row::new(vec!["*", "multiply"]),
        Row::new(vec!["/", "divide"]),
        Row::new(vec!["\\", "invert (1/x)"]),
        Row::new(vec!["^", "power"]),
        Row::new(vec!["L", "ln"]),
        Row::new(vec!["V", "square root"]),
        Row::new(vec!["A", "absolute"]),
        Row::new(vec!["%", "modulo"]),
        Row::new(vec!["R", "remainder"]),
        Row::new(vec!["S", "sine"]),
        Row::new(vec!["C", "cosine"]),
        Row::new(vec!["T", "tangent"]),
        Row::new(vec!["~", "round"]),
        Row::new(vec!["D", "round down"]),
        Row::new(vec!["U", "round up"]),
    ])
    .header(header)
    .column_spacing(1)
    .widths([Constraint::Length(8), Constraint::Percentage(50)].as_ref());

    f.render_widget(right_table, root[1]);
}

/// Draws the default screen interface.
fn draw_default_screen(state: &State, f: &mut Frame<CrosstermBackend<io::Stdout>>) {
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
                Constraint::Length(27),
                Constraint::Percentage(50),
            ]
            .as_ref(),
        )
        .split(root[0]);

    let centre_stack = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(top_section[1]);

    let stack_box = List::new(
        state
            .calc_state
            .stack_vec()
            .iter()
            .rev()
            .map(|i| ListItem::new(format!("{i}")))
            .collect::<Vec<ListItem>>(),
    )
    .start_corner(tui::layout::Corner::BottomLeft)
    .block(Block::default().title("Stack").borders(Borders::ALL));
    f.render_widget(stack_box, top_section[0]);

    let register_box = List::new(
        state
            .calc_state
            .registers_vec()
            .iter()
            .map(|(k, v)| ListItem::new(format!("{k}: {v}")))
            .collect::<Vec<ListItem>>(),
    )
    .block(Block::default().title("Registers").borders(Borders::ALL));
    f.render_widget(register_box, centre_stack[0]);

    let stack_top = if let Some(rpn::Num::Int(n)) = state.calc_state.stack_last() {
        *n
    } else {
        0
    };
    let stack_size = state.calc_state.stack_size();

    let info_box = Paragraph::new(format!(
        "Status: Fully operational
Input mode:           Dec
Stack size: {stack_size:>13}
─────────────────────────
Bin: {stack_top:>20b}
Oct: {stack_top:>20o}
Dec: {stack_top:>20}
Hex: {stack_top:>20x}"
    ))
    .block(Block::default().title("Info").borders(Borders::ALL));
    f.render_widget(info_box, centre_stack[1]);

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

    let input_box = Paragraph::new(state.message.clone().unwrap_or_else(|| state.input.clone()))
        .block(Block::default().title("Input").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    f.render_widget(input_box, root[1]);
}

/// Formats a history event for display to the user.
fn format_history_event(state: &rpn::State, op: &Op) -> String {
    let stack_size = state.stack_size();
    match op {
        Op::Push(n) => format!("-> {n}"),
        Op::Rotate => ">>>".to_string(),
        Op::Swap => format!(
            "{} <-> {}",
            state.stack_get(stack_size - 2).unwrap(),
            state.stack_get(stack_size - 1).unwrap()
        ),
        Op::Drop => format!("<- {}", state.stack_last().unwrap()),
        Op::Negate => format!("(-) {}", state.stack_last().unwrap()),
        Op::Clear => "- clear -".to_string(),
        Op::Store(k) => format!("store {} -> {k}", state.stack_last().unwrap()),
        Op::Recall(k) => format!("recall {k}"),
        Op::Add => format!(
            "{} + {}",
            state.stack_get(stack_size - 2).unwrap(),
            state.stack_get(stack_size - 1).unwrap()
        ),
        Op::Subtract => format!(
            "{} - {}",
            state.stack_get(stack_size - 2).unwrap(),
            state.stack_get(stack_size - 1).unwrap()
        ),
        Op::Multiply => format!(
            "{} × {}",
            state.stack_get(stack_size - 2).unwrap(),
            state.stack_get(stack_size - 1).unwrap()
        ),
        Op::Divide => format!(
            "{} / {}",
            state.stack_get(stack_size - 2).unwrap(),
            state.stack_get(stack_size - 1).unwrap()
        ),
        Op::Pow => format!(
            "{} ^ {}",
            state.stack_get(stack_size - 2).unwrap(),
            state.stack_get(stack_size - 1).unwrap()
        ),
        Op::Absolute => format!("|{}|", state.stack_last().unwrap()),
        Op::Modulo => format!(
            "{} mod {}",
            state.stack_get(stack_size - 2).unwrap(),
            state.stack_get(stack_size - 1).unwrap()
        ),
        Op::Remainder => format!(
            "{} rem {}",
            state.stack_get(stack_size - 2).unwrap(),
            state.stack_get(stack_size - 1).unwrap()
        ),
        Op::Round => format!("≈ {}", state.stack_last().unwrap()),
        Op::Floor => format!("↓ {}", state.stack_last().unwrap()),
        Op::Ceiling => format!("↑ {}", state.stack_last().unwrap()),
        Op::Sine => format!("sin {}", state.stack_last().unwrap()),
        Op::Cosine => format!("cos {}", state.stack_last().unwrap()),
        Op::Tangent => format!("tan {}", state.stack_last().unwrap()),
        Op::SquareRoot => format!("√ {}", state.stack_last().unwrap()),
        Op::NaturalLogarithm => format!("ln {}", state.stack_last().unwrap()),
        Op::Invert => format!("1 / {}", state.stack_last().unwrap()),
        _ => format!("{state:?} -> {op:?}"),
    }
}
