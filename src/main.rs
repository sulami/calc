mod rpn;

fn main() {
    let s = rpn::State::default();
    let ops = &[
        rpn::Op::Push(1.into()),
        rpn::Op::Push(2.into()),
        rpn::Op::Push(3.into()),
        rpn::Op::Swap,
        rpn::Op::Drop,
        rpn::Op::Rotate,
        rpn::Op::Add,
        rpn::Op::Push(4.into()),
        rpn::Op::Subtract,
    ];
    println!("{:?}", rpn::run(s, ops));
}
