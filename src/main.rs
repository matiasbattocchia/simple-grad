use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use std::iter::zip;

type Value = i32;

type Var = Rc<Variable>;

struct Variable {
    value: Value,
    name: String,
    grad: RefCell<Option<Var>>,
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Variable {
    fn grad(&self) -> Option<Var> {
        if let Some(grad) = self.grad.borrow().as_ref() {
            Some( Rc::clone(grad) )
        } else {
            None
        }
    }
}

#[derive(Clone)]
struct TapeEntry {
    inputs: Vec<String>,
    output: String,
    backprop: Rc<dyn Fn(&Ops, &[&Var]) -> Vec<Var>>,
}

struct Ops {
    tape: RefCell<Vec<TapeEntry>>,
    counter: RefCell<u8>,
}

impl Ops {
    fn new() -> Self {
        Ops{ tape: RefCell::new(vec![]), counter: RefCell::new(0) }
    }

    fn var(&self, value: Value) -> Var {
        let v = Rc::new( Variable{ value, name: String::from( format!("v{}", self.counter.borrow()) ), grad: RefCell::new(None) } );

        *self.counter.borrow_mut() += 1;

        v
    }

    fn named_var(&self, value: Value, name: &str) -> Var {
        let v = Rc::new( Variable{ value, name: name.to_string(), grad: RefCell::new(None) } );
        println!("{} = {}", v.name, v.value);

        v
    }

    fn mul(&self, a: &Var, b: &Var) -> Var {
        let a = Rc::clone(a);
        let b = Rc::clone(b);

        let inputs = vec![a.name.clone(), b.name.clone()];

        let r = self.var(a.value * b.value);
        println!("{} = {} * {}", r.name, a.name, b.name);

        let backprop = move | op: &Ops, dl_dr: &[&Var] | {
            let dr_da = &b;
            let dr_db = &a;

            let dl_da = op.mul(dl_dr[0], dr_da);
            let dl_db = op.mul(dl_dr[0], dr_db);

            vec![dl_da, dl_db]
        };

        self.tape.borrow_mut().push(TapeEntry{
            inputs,
            output: r.name.clone(),
            backprop: Rc::new(backprop),
        });

        r
    }

    fn add(&self, a: &Var, b: &Var) -> Var {
        let a = Rc::clone(a);
        let b = Rc::clone(b);

        let inputs = vec![a.name.clone(), b.name.clone()];

        let r = self.var(a.value + b.value);
        println!("{} = {} + {}", r.name, a.name, b.name);

        let backprop = move | _op: &Ops, dl_dr: &[&Var] | {
            // dr_da = 1;
            // dr_db = 1;

            let dl_da = Rc::clone(dl_dr[0]); // * dr_da
            let dl_db = Rc::clone(dl_dr[0]); // * dr_db

            vec![dl_da, dl_db]
        };

        self.tape.borrow_mut().push(TapeEntry{
            inputs,
            output: r.name.clone(),
            backprop: Rc::new(backprop),
        });

        r
    }

    fn grad(&self, l: &Var, ret: &[&Var]) {
        println!("d{} --------------", l.name);
        let mut dl_d = HashMap::new();
        dl_d.insert(l.name.clone(), self.var(1));

        // cannot borrow for iteration and then borrow mutable during backprop
        let tape = self.tape.borrow().clone();

        for entry in tape.iter().rev() {
            //println!("{:?} -> {:?}", entry.inputs, entry.output);

            let dl_doutput = &dl_d.get(&entry.output);

            if dl_doutput.is_none() { continue }

            // dl_doutput is a single element slice by the time being; tape entry should support
            // many outputs
            let dl_dinputs = (entry.backprop)(self, &[dl_doutput.unwrap()]);

            for (input, dl_dinput) in zip(&entry.inputs, dl_dinputs) {
                if let Some(grad) = dl_d.get(input) {
                    dl_d.insert(input.clone(), self.add(&grad, &dl_dinput));
                } else {
                    dl_d.insert(input.clone(), dl_dinput);
                }
            }
        }

        for (key, value) in &dl_d {
            println!("d{}_d{} = {}", l.name, key, value.name);
        }
        println!("------------------");

        for var in ret {
            *var.grad.borrow_mut() = dl_d.remove(&var.name);
        }
    }
}

fn main() {
    let op = Ops::new();

    let a = op.named_var(2, "a");
    let b = op.named_var(3, "b");

    let t = op.add(&a, &b);
    let l = op.mul(&t, &b);

    op.grad(&l, &[&a, &b]);

    let da = a.grad().unwrap();
    let db = b.grad().unwrap();

    println!("d{} = {}", a.name, da);
    println!("d{} = {}", b.name, db);

    let l = op.mul(&da, &db);
    op.grad(&l, &[&a, &b]);

    let da = a.grad().unwrap();
    let db = b.grad().unwrap();

    println!("d{} = {}", a.name, da);
    println!("d{} = {}", b.name, db);
}
