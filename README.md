# Simple Grad

This work is a self-contained implementation of reverse mode automatic differentiation,
based on that of PyTorch's and made way more simpler with the intention to understand the basics
of how it works.

This is a Rust implementation of the code I found in
[A Gentle Introduction to `torch.autograd`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
tutorial, below further readings, a notebook named
[Example implementation of reverse-mode autodiff](https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC).

## Educational project

This is my first Rust project. Having read 55% of the book I felt I needed to put my readings into practice following these principles:
1. Implement an existing program with few modifications as possible, so I could focus in Rust.
2. Do not look other implementations in beforehand. There are plenty of Rust autograds projects outhere, innocent inspiration would have ended in shameless
no-brain copy and paste.
3. Do not improve it further. Transforming this development into a deep learning framework is tempting, not only there are better ones but I shall not forget
the main reason to start it: to practice some Rust.

## It works

It can take gradients of gradients.

```rust
let op = Ops::new();

let a = op.named_var(Tensor::new(&[0.4605, 0.4061, 0.9422, 0.3946], &Device::Cpu).unwrap(), "a");
let b = op.named_var(Tensor::new(&[0.0850, 0.3296, 0.9888, 0.6494], &Device::Cpu).unwrap(), "b");

let simple = | a, b | { op.mul( &op.add(a, b), b ) };

let l0 = op.sum( &simple(&a, &b) , Some("L0") );
op.grad(&l0, &[&a, &b]);

let dl0_da = a.grad().unwrap();
let dl0_db = b.grad().unwrap();

let l1 = op.sum( &op.add( &op.mul(&dl0_da, &dl0_da), &op.mul(&dl0_db, &dl0_db) ), Some("L1") );
op.grad(&l1, &[&a, &b]);

let dl1_da = a.grad().unwrap();
let dl1_db = b.grad().unwrap();

println!("d{} = {}", a.name, dl1_da);
println!("d{} = {}", b.name, dl1_db);
```

Output:

```
a = [0.4605, 0.4061, 0.9422, 0.3946]
b = [0.0850, 0.3296, 0.9888, 0.6494]
v0 = a + b
v1 = v0 * b
L0 = v1.sum()
dL0 --------------
v3 = v2.expand([4])
v4 = v3 * b
v5 = v3 * v0
v6 = v5 + v4
dL0_dv1 = v3
dL0_dv0 = v4
dL0_dL0 = v2
dL0_da = v4
dL0_db = v6
------------------
v7 = v4 * v4
v8 = v6 * v6
v9 = v7 + v8
L1 = v9.sum()
dL1 --------------
v11 = v10.expand([4])
v12 = v11 * v6
v13 = v11 * v6
v14 = v12 + v13
v15 = v11 * v4
v16 = v11 * v4
v17 = v15 + v16
v18 = v17 + v14
v19 = v14 * v0
v20 = v14 * v3
v21 = v18 * b
v22 = v18 * v3
v23 = v19 + v21
v24 = v23.sum()
v25 = v22 + v20
dL1_dv0 = v20
dL1_dv4 = v18
dL1_dv2 = v24
dL1_dv5 = v14
dL1_dv9 = v11
dL1_dL1 = v10
dL1_db = v25
dL1_dv6 = v14
dL1_dv3 = v23
dL1_dv7 = v11
dL1_da = v20
dL1_dv8 = v11
------------------
da = [1.2610, 2.1306, 5.8396, 3.3868]
db = [ 2.6920,  4.9204, 13.6568,  8.0724]
```

## Design decisions

I wanted to support syntax like the following (i.e. not enforcing binding of returned variables).

```rust
let y = op.mul( &op.add(&a, &b), &b );
```

I did not bother in making operations as variable operators, a functional approach would be enough for the time being.

It should work with tensor or ndarray data types which are expected to not support the `Copy` trait and which might to be expensive to `clone()`.

```rust
type Var = Rc<Variable>;

struct Variable {
    value: Value,
    name: String,
    grad: RefCell<Option<Var>>,
}
```

So I put variables inside a `Rc<T>`.

Some people store a reference to the tape into de variables, others, choose to do it into operations, not to mentions the ones who prefer a global variable.
I did it in operations because it was the thing that worked the best for my current Rust knowledge.

```rust
struct Ops {
    tape: RefCell<Vec<TapeEntry>>,
    counter: RefCell<u8>,
}
```

I had to use `RefCell<T>` for the tape because there might be many references to Ops and they have to be mutable.
