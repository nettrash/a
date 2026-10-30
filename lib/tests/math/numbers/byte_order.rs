use a::math::numbers::ByteOrder;

#[test]
fn byte_orders_equal_themselves() {
    assert!(ByteOrder::BigEndian == ByteOrder::BigEndian);
    assert!(ByteOrder::LittleEndian == ByteOrder::LittleEndian);
}

#[test]
fn byte_orders_differ() {
    assert!(ByteOrder::BigEndian != ByteOrder::LittleEndian);
    assert!(ByteOrder::LittleEndian != ByteOrder::BigEndian);
}

#[test]
fn byte_order_is_copy() {
    let order = ByteOrder::LittleEndian;
    let copy = order;
    #[allow(clippy::clone_on_copy)]
    let clone = order.clone();
    // `order` is still usable after being copied.
    assert!(order == copy);
    assert!(order == clone);
}
