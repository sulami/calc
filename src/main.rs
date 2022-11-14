mod rpn;

fn main() {
    println!("Hello, world!");
    let s = rpn::State::default();
    let ops = &[
        rpn::Op::Push(1.into()),
        rpn::Op::Push(2.into()),
        rpn::Op::Push(3.into()),
        rpn::Op::Swap,
    ];
    println!("{:?}", rpn::run(s, ops));
}
