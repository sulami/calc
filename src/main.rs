use std::io;

use anyhow::Result;
use copypasta::{ClipboardContext, ClipboardProvider};
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
    mode: Mode,
    base: Base,
    input: String,
    message: Option<String>,
}

/// The different main modes the app can be in. Defines the display
/// and keybindings.
#[derive(Copy, Clone)]
enum Mode {
    Normal,
    StoreRegister,
    RecallRegister,
    Help,
    Exit,
}

impl Default for Mode {
    fn default() -> Self {
        Self::Normal
    }
}

/// The different number bases for input and display.
#[derive(Copy, Clone)]
enum Base {
    Binary,
    Octal,
    Decimal,
    Hex,
}

impl Default for Base {
    fn default() -> Self {
        Self::Decimal
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
    // Remove "hit h for help" on first re-render.
    state.message = None;

    loop {
        match state.mode {
            Mode::Normal => {
                state = normal_mode_handler(state, read()?)?;
                terminal.draw(|f| draw_ui(&state, f))?;

                if state.message.is_some() {
                    state.message = None;
                }
            }
            Mode::StoreRegister => {
                state = store_mode_handler(state, read()?)?;
                terminal.draw(|f| draw_ui(&state, f))?;
            }
            Mode::RecallRegister => {
                state = recall_mode_handler(state, read()?)?;
                terminal.draw(|f| draw_ui(&state, f))?;
            }
            Mode::Help => {
                state = help_mode_handler(state, read()?)?;
                terminal.draw(|f| draw_ui(&state, f))?;
            }
            Mode::Exit => return Ok(()),
        }
    }
}

/// Input handler for regular calculator operation.
fn normal_mode_handler(mut state: State, event: Event) -> Result<State> {
    match event {
        Event::Key(KeyEvent {
            code: KeyCode::Char(c),
            ..
        }) => match c {
            '0'..='9' | 'a'..='f' => match (state.base, c) {
                (_, '0'..='1') => state.input.push(c),
                (Base::Octal | Base::Decimal | Base::Hex, '2'..='8') => state.input.push(c),
                (Base::Decimal | Base::Hex, '9') => state.input.push(c),
                (Base::Hex, _) => state.input.push(c),
                _ => (),
            },
            '.' => {
                if let Base::Decimal = state.base {
                    if !state.input.contains('.') {
                        state.input.push(c)
                    }
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
            'h' => state.mode = Mode::Help,
            'z' => state = try_op(state, Op::Drop),
            'L' => {
                state = insert_input(state);
                state = try_op(state, Op::NaturalLogarithm);
            }
            '_' => {
                state = insert_input(state);
                state = try_op(state, Op::Logarithm);
            }
            'n' => {
                state = insert_input(state);
                state = try_op(state, Op::Negate);
            }
            'P' => state = try_op(state, Op::Push(std::f64::consts::PI.into())),
            'q' => state = try_op(state, Op::Rotate),
            'i' => {
                state = insert_input(state);
                state = try_op(state, Op::IntegerDivision);
            }
            'w' => state = try_op(state, Op::Swap),
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
            'k' => {
                state = insert_input(state);
                state.message = Some("enter register name".to_string());
                state.mode = Mode::StoreRegister;
            }
            'l' => {
                state = insert_input(state);
                state.message = Some("enter register name".to_string());
                state.mode = Mode::RecallRegister;
            }
            'y' => {
                state = insert_input(state);
                let stack_top: rpn::Num = *state.calc_state.stack_last().unwrap_or(&0.into());
                let formatted = format_num(&stack_top, state.base);
                state.message = Some(format!("yanked: {formatted}"));
                ClipboardContext::new()
                    .expect("failed to create clipboard context")
                    .set_contents(formatted)
                    .expect("failed to insert clipboard contents");
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
            '<' => {
                state = insert_input(state);
                state = try_op(state, Op::ShiftLeft);
            }
            '>' => {
                state = insert_input(state);
                state = try_op(state, Op::ShiftRight);
            }
            '|' => {
                state = insert_input(state);
                state = try_op(state, Op::BitwiseOr);
            }
            '&' => {
                state = insert_input(state);
                state = try_op(state, Op::BitwiseAnd);
            }
            'j' => {
                state = insert_input(state);
                state = try_op(state, Op::BitwiseXor);
            }
            '#' => state = try_op(state, Op::Push(rand::random::<f64>().into())),
            'Q' => state.mode = Mode::Exit,
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
            KeyCode::Tab => {
                // NB Avoid having input with out of range numbers
                // buffered in input.
                state.input.clear();
                state.base = match state.base {
                    Base::Binary => Base::Octal,
                    Base::Octal => Base::Decimal,
                    Base::Decimal => Base::Hex,
                    Base::Hex => Base::Binary,
                };
            }
            _ => (),
        },
        _ => (),
    }

    Ok(state)
}

/// Input handler for store register.
fn store_mode_handler(mut state: State, event: Event) -> Result<State> {
    if let Event::Key(KeyEvent {
        code: KeyCode::Char(c),
        ..
    }) = event
    {
        state.mode = Mode::Normal;
        state.message = None;
        match c {
            'a'..='z' => state = try_op(state, Op::Store(c)),
            'A'..='Z' => state = try_op(state, Op::Store(c)),
            _ => state.message = Some("invalid register name".to_string()),
        }
    }
    Ok(state)
}

/// Input handler for recall register.
fn recall_mode_handler(mut state: State, event: Event) -> Result<State> {
    if let Event::Key(KeyEvent {
        code: KeyCode::Char(c),
        ..
    }) = event
    {
        state.mode = Mode::Normal;
        state.message = None;
        match c {
            'a'..='z' => state = try_op(state, Op::Recall(c)),
            'A'..='Z' => state = try_op(state, Op::Recall(c)),
            _ => state.message = Some("invalid register name".to_string()),
        }
    }
    Ok(state)
}

/// Input handler for help mode.
fn help_mode_handler(mut state: State, event: Event) -> Result<State> {
    if let Event::Key(KeyEvent {
        code: KeyCode::Char('h'),
        ..
    }) = event
    {
        state.mode = Mode::Normal;
    }

    Ok(state)
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
        let base = match state.base {
            Base::Binary => 2,
            Base::Octal => 8,
            Base::Decimal => 10,
            Base::Hex => 16,
        };
        let num = i128::from_str_radix(&state.input, base)
            .map(rpn::Num::from)
            .ok()
            .or_else(|| -> Option<rpn::Num> {
                if let Base::Decimal = state.base {
                    state.input.parse::<f64>().map(rpn::Num::from).ok()
                } else {
                    None
                }
            })
            .expect("invalid number");
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
        Mode::Help => draw_help_screen(f),
        _ => draw_default_screen(state, f),
    }
}

/// Draws the help screen interface.
fn draw_help_screen(f: &mut Frame<CrosstermBackend<io::Stdout>>) {
    let root = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(f.size());

    let header = Row::new(vec![Cell::from("Key"), Cell::from("Action")]).bottom_margin(1);

    let left_table = Table::new(vec![
        Row::new(vec!["0-9, a-f", "number input"]),
        Row::new(vec!["enter", "push to stack"]),
        Row::new(vec!["tab", "switch input mode"]),
        Row::new(vec!["u", "undo"]),
        Row::new(vec!["h", "toggle help screen"]),
        Row::new(vec!["Q", "quit"]).bottom_margin(1),
        Row::new(vec!["z", "drop"]),
        Row::new(vec!["w", "swap"]),
        Row::new(vec!["q", "rotate stack"]).bottom_margin(1),
        Row::new(vec!["k <key>", "store to register"]),
        Row::new(vec!["l <key>", "recall from register"]).bottom_margin(1),
        Row::new(vec!["y", "yank to clipboard"]).bottom_margin(1),
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
        Row::new(vec!["_", "logarithm base x"]),
        Row::new(vec!["V", "square root"]),
        Row::new(vec!["A", "absolute"]),
        Row::new(vec!["%", "modulo"]),
        Row::new(vec!["i", "integer division"]),
        Row::new(vec!["S", "sine"]),
        Row::new(vec!["C", "cosine"]),
        Row::new(vec!["T", "tangent"]),
        Row::new(vec!["~", "round"]),
        Row::new(vec!["U / D", "round up/down"]),
        Row::new(vec!["< / >", "shift left/right"]),
        Row::new(vec!["&", "bitwise and"]),
        Row::new(vec!["|", "bitwise or"]),
        Row::new(vec!["j", "bitwise xor"]),
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
            .map(|i| {
                ListItem::new(format_num_ralign(
                    i,
                    state.base,
                    top_section[0].width as usize - 2,
                ))
            })
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
            .map(|(k, v)| {
                ListItem::new(format!(
                    "{k}: {}",
                    format_num_ralign(v, state.base, centre_stack[0].width as usize - 5)
                ))
            })
            .collect::<Vec<ListItem>>(),
    )
    .block(Block::default().title("Registers").borders(Borders::ALL));
    f.render_widget(register_box, centre_stack[0]);

    let stack_top: rpn::Num = *state.calc_state.stack_last().unwrap_or(&0.into());
    let stack_size = state.calc_state.stack_size();
    let base = match state.base {
        Base::Binary => "bin",
        Base::Octal => "oct",
        Base::Decimal => "dec",
        Base::Hex => "hex",
    };
    let num_type = match stack_top {
        rpn::Num::Int(_) => "integer",
        rpn::Num::Float(_) => "float",
    };

    let info_box = Paragraph::new(format!(
        "Input mode: {base:>13}
Stack size: {stack_size:>13}
─────────────────────────
Type: {num_type:>19}
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
            .map(|(op, s)| ListItem::new(format_history_event(s, op, state.base)))
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

/// Formats a number based on the base selected. Right-aligns to the
/// provided width.
fn format_num_ralign(n: &rpn::Num, base: Base, width: usize) -> String {
    match base {
        Base::Binary => format!("{n:>width$b}", width = width),
        Base::Octal => format!("{n:>width$o}", width = width),
        Base::Decimal => format!("{n:>width$}", width = width),
        Base::Hex => format!("{n:>width$x}", width = width),
    }
}

/// Formats a number based on the base selected.
fn format_num(n: &rpn::Num, base: Base) -> String {
    match base {
        Base::Binary => format!("{n:b}"),
        Base::Octal => format!("{n:o}"),
        Base::Decimal => format!("{n}"),
        Base::Hex => format!("{n:x}"),
    }
}

/// Formats a history event for display to the user.
fn format_history_event(state: &rpn::State, op: &Op, base: Base) -> String {
    let stack_size = state.stack_size();
    match op {
        Op::Push(n) => format!("-> {}", format_num(n, base)),
        Op::Rotate => "⭮".to_string(),
        Op::Swap => format!(
            "{} <-> {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base),
        ),
        Op::Drop => format!("<- {}", format_num(state.stack_last().unwrap(), base)),
        Op::Negate => format!("(-) {}", format_num(state.stack_last().unwrap(), base)),
        Op::Clear => "- clear -".to_string(),
        Op::Store(k) => format!(
            "store {} -> {k}",
            format_num(state.stack_last().unwrap(), base)
        ),
        Op::Recall(k) => format!("recall {k}"),
        Op::Add => format!(
            "{} + {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
        Op::Subtract => format!(
            "{} - {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
        Op::Multiply => format!(
            "{} × {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
        Op::Divide => format!(
            "{} / {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
        Op::Pow => format!(
            "{} ^ {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
        Op::Absolute => format!("|{}|", format_num(state.stack_last().unwrap(), base)),
        Op::Modulo => format!(
            "{} mod {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
        Op::IntegerDivision => format!(
            "{} // {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
        Op::Round => format!("≈ {}", format_num(state.stack_last().unwrap(), base)),
        Op::Floor => format!("↓ {}", format_num(state.stack_last().unwrap(), base)),
        Op::Ceiling => format!("↑ {}", format_num(state.stack_last().unwrap(), base)),
        Op::Sine => format!("sin {}", format_num(state.stack_last().unwrap(), base)),
        Op::Cosine => format!("cos {}", format_num(state.stack_last().unwrap(), base)),
        Op::Tangent => format!("tan {}", format_num(state.stack_last().unwrap(), base)),
        Op::SquareRoot => format!("√ {}", format_num(state.stack_last().unwrap(), base)),
        Op::NaturalLogarithm => format!("ln {}", format_num(state.stack_last().unwrap(), base)),
        Op::Logarithm => format!(
            "log{{{}}}({})",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
        Op::Invert => format!("1 / {}", format_num(state.stack_last().unwrap(), base)),
        Op::ShiftLeft => format!("{} << 1", format_num(state.stack_last().unwrap(), base)),
        Op::ShiftRight => format!("{} >> 1", format_num(state.stack_last().unwrap(), base)),
        Op::BitwiseAnd => format!(
            "{} & {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
        Op::BitwiseOr => format!(
            "{} | {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
        Op::BitwiseXor => format!(
            "{} ⊕ {}",
            format_num(state.stack_get(stack_size - 2).unwrap(), base),
            format_num(state.stack_get(stack_size - 1).unwrap(), base)
        ),
    }
}
