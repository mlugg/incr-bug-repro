pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const source = try std.fs.cwd().readFileAllocOptions(gpa, "source.zig", 1 << 32, null, 1, 0);
    defer gpa.free(source);

    var tree = try Ast.parse(gpa, source, .zig);
    defer tree.deinit(gpa);

    var zir = try ZirGen.generate(gpa, &tree);
    defer zir.deinit(gpa);

    if (zir.compile_errors.len > 0) {
        var wip_errors: std.zig.ErrorBundle.Wip = undefined;
        try wip_errors.init(gpa);
        defer wip_errors.deinit();
        try addZirErrorMessages(&wip_errors, &zir, tree, source, "source.zig");
        var eb = try wip_errors.toOwnedBundle("");
        defer eb.deinit(gpa);
        eb.renderToStdErr(.{ .ttyconf = .escape_codes });
    }
    if (zir.instructions.len > 0) {
        try print_zir.dump(std.io.getStdOut().writer(), &zir);
    }
}

pub fn addZirErrorMessages(
    eb: *std.zig.ErrorBundle.Wip,
    zir: *const Zir,
    tree: std.zig.Ast,
    source: [:0]const u8,
    src_path: []const u8,
) !void {
    assert(zir.compile_errors.len > 0);

    for (zir.compile_errors) |err| {
        const err_span: std.zig.Ast.Span = switch (err.src.unwrap()) {
            .token_and_offset => |tao| span: {
                const token_start = tree.tokenStart(tao.token);
                const start = token_start + tao.byte_offset;
                const end = token_start + @as(u32, @intCast(tree.tokenSlice(tao.token).len));
                break :span .{ .start = start, .end = end, .main = start };
            },
            .node => |node| tree.nodeToSpan(node),
        };
        const err_loc = std.zig.findLineColumn(source, err_span.main);

        try eb.addRootErrorMessage(.{
            .msg = try eb.addString(err.msg.get(zir)),
            .src_loc = try eb.addSourceLocation(.{
                .src_path = try eb.addString(src_path),
                .span_start = err_span.start,
                .span_main = err_span.main,
                .span_end = err_span.end,
                .line = @intCast(err_loc.line),
                .column = @intCast(err_loc.column),
                .source_line = try eb.addString(err_loc.source_line),
            }),
            .notes_len = err.note_count,
        });

        const notes_start = try eb.reserveNotes(err.note_count);
        for (notes_start.., err.getNotes(zir), 0..err.note_count) |eb_note_idx, note, _| {
            const note_span: std.zig.Ast.Span = switch (note.src.unwrap()) {
                .token_and_offset => |tao| span: {
                    const token_start = tree.tokenStart(tao.token);
                    const start = token_start + tao.byte_offset;
                    const end = token_start + @as(u32, @intCast(tree.tokenSlice(tao.token).len));
                    break :span .{ .start = start, .end = end, .main = start };
                },
                .node => |node| tree.nodeToSpan(node),
            };
            const note_loc = std.zig.findLineColumn(source, note_span.main);

            // This line can cause `wip.extra.items` to be resized.
            const note_index = @intFromEnum(try eb.addErrorMessage(.{
                .msg = try eb.addString(note.msg.get(zir)),
                .src_loc = try eb.addSourceLocation(.{
                    .src_path = try eb.addString(src_path),
                    .span_start = note_span.start,
                    .span_main = note_span.main,
                    .span_end = note_span.end,
                    .line = @intCast(note_loc.line),
                    .column = @intCast(note_loc.column),
                    .source_line = if (note_loc.eql(err_loc))
                        0
                    else
                        try eb.addString(note_loc.source_line),
                }),
                .notes_len = 0,
            }));
            eb.extra.items[eb_note_idx] = note_index;
        }
    }
}

const ZirGen = @import("ZirGen.zig");
const Zir = @import("Zir.zig");
const print_zir = @import("print_zir.zig");
const std = @import("std");
const Ast = std.zig.Ast;
const assert = std.debug.assert;
