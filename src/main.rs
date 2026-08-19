mod client;
mod controller;
mod permissions;

fn main() {
    let x: i32 = 42;
    let y = Box::new(42);

    println!("x = {}, y = {}", x, *y);
}
