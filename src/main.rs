use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

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
    inputs: (String, String),
    output: String,
    backprop: Rc<dyn Fn(&Ops, &Var) -> (Var,Var)>,
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

        let inputs = (a.name.clone(), b.name.clone());

        let r = self.var(a.value * b.value);
        println!("{} = {} * {}", r.name, a.name, b.name);

        let backprop = move | op: &Ops, dl_dr: &Var | {
            let dr_da = &b;
            let dr_db = &a;

            let dl_da = op.mul(dl_dr, dr_da);
            let dl_db = op.mul(dl_dr, dr_db);

            (dl_da, dl_db)
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

        let inputs = (a.name.clone(), b.name.clone());

        let r = self.var(a.value + b.value);
        println!("{} = {} + {}", r.name, a.name, b.name);

        let backprop = move | _op: &Ops, dl_dr: &Var | {
            // dr_da = 1;
            // dr_db = 1;

            let dl_da = Rc::clone(dl_dr); // * dr_da
            let dl_db = Rc::clone(dl_dr); // * dr_db

            (dl_da, dl_db)
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
            println!("{:?} -> {:?}", entry.inputs, entry.output);

            let dl_doutput = &dl_d.get(&entry.output);

            if dl_doutput.is_none() { continue }

            let dl_dinputs = (entry.backprop)(self, dl_doutput.unwrap());

            if let Some(grad) = dl_d.get(&entry.inputs.0) {
                dl_d.insert(entry.inputs.0.clone(), self.add(&grad, &dl_dinputs.0));
            } else {
                dl_d.insert(entry.inputs.0.clone(), dl_dinputs.0);
            }

            if let Some(grad) = dl_d.get(&entry.inputs.1) {
                dl_d.insert(entry.inputs.1.clone(), self.add(&grad, &dl_dinputs.1));
            } else {
                dl_d.insert(entry.inputs.1.clone(), dl_dinputs.1);
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
