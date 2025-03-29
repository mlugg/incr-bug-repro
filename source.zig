// top-level fields
some_field: u32 = 123,

// declarations and type annotations
pub const foo = 123;
var bar: u32 = foo; // referencing a decl

threadlocal const x;

comptime {
    const something = b: {
        break :b 123;
        456; // statically unreachable code
    };
    @import("foo.zig"); // this import is added to a list of all imports
    @import("bar.zig");
    @import("foo.zig"); // (which is deduped)
}

// fatal zirgen error lowers to `zirgen_error` instruction
const another: u32 = @bad();

usingnamespace 42 + 1; // 1 is @one, not an instruction

test {
    123; // unnamed test
}


test "hi there" {
    456; // named test
}

test foo {
    789; // valid decltest
    123.456ee0;
}

test aBadIdentifier {
    4242; // invalid decltest
    "hello, world!";
    "hello, world!";
    "not\x00deduped";
    "not\x00deduped";
}

test "numbers" {
    1;
    0;
    -1;
    10;
    -10;
    132473219857439785647564785685674389756356378456;
    -132473219857439785647564785685674389756356378456;
    123.45;
    -123.45;
    @as(u8, 1);
    @as(u16, 1);
    @as(u32, 1);
    @as(usize, 1);
}

fn fooFunc(comptime T: type, x: T) T {
    _ = x;
    return &x;
}
