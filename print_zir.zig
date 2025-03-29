pub fn dump(w: anytype, zir: *const Zir) !void {
    try w.print(
        \\# Total ZIR Bytes: {}
        \\# Instructions: {} ({})
        \\# Extra Data Items: {} ({})
        \\# String Data: {}
        \\
    , .{
        std.fmt.fmtIntSizeBin(zir.instructions.len * 9 + zir.extra.len * 4 + zir.string_bytes.len),
        zir.instructions.len,
        std.fmt.fmtIntSizeBin(zir.instructions.len * 9),
        zir.extra.len,
        std.fmt.fmtIntSizeBin(zir.extra.len * 4),
        std.fmt.fmtIntSizeBin(zir.string_bytes.len),
    });
    try w.writeAll("# Imports: ");
    if (zir.imports.len == 0) {
        try w.writeAll("(none)");
    } else for (zir.imports, 0..) |import_str, idx| {
        if (idx != 0) try w.writeAll(", ");
        try w.print("\"{}\"", .{std.zig.fmtEscapes(import_str.get(zir))});
    }
    try w.writeAll("\n\n");
    try dumpInner(w, 0, zir, @enumFromInt(0));
}

fn dumpInner(w: anytype, indent: u32, zir: *const Zir, inst_idx: Zir.Inst.Index) (@TypeOf(w).Error)!void {
    try w.writeByteNTimes(' ', indent);
    try w.print("%{d} = ", .{@intFromEnum(inst_idx)});
    const inst = inst_idx.get(zir);
    switch (inst) {
        .int => |x| try w.print("int({d})", .{x}),
        .int_big => |x| try w.print("int_big({d})", .{x.get(zir)}),
        .float => |f| try w.print("float({})", .{f}),
        .str => |s| try w.print("str(\"{}\")", .{std.zig.fmtEscapes(s.get(zir))}),

        .zirgen_error,
        .ret_implicit,
        => try w.print("{s}()", .{@tagName(inst)}),

        .add,
        .add_wrap,
        .add_sat,
        .sub,
        .sub_wrap,
        .sub_sat,
        .mul,
        .mul_wrap,
        .mul_sat,
        .div,
        .mod_rem,
        .shl_sat,
        => |bin| try w.print("{s}({}, {})", .{ @tagName(inst), fmtRef(bin.lhs), fmtRef(bin.rhs) }),

        .coerce => |c| try w.print("coerce({}, {})", .{ fmtRef(c.operand), fmtRef(c.dest_ty) }),
        .store => |s| try w.print("store({}, {})", .{ fmtRef(s.ptr), fmtRef(s.operand) }),
        .import => |i| try w.print("import({}, \"{}\")", .{ fmtRef(i.res_ty), std.zig.fmtEscapes(i.path.get(zir)) }),

        .validate_const,
        .validate_const_ref,
        .ref,
        .ensure_result_used,
        .load,
        .negate,
        .ret,
        => |op| try w.print("{s}({})", .{ @tagName(inst), fmtRef(op) }),

        .decl_ref => |name| try w.print("decl_ref(\"{}\")", .{std.zig.fmtEscapes(name.get(zir))}),

        .block => |block| {
            try w.writeAll("block(");
            try dumpBody(w, indent, zir, block.body);
            try w.writeByte(')');
        },
        .@"break" => |br| try w.print("break(%{d}, {})", .{ @intFromEnum(br.target), fmtRef(br.operand) }),
        .struct_decl => |struct_decl| {
            try w.print("struct_decl(line={d}, hash=TODO, captures=TODO, backing=TODO, decls=", .{
                struct_decl.src_line,
            });
            try printDecls(w, indent, zir, struct_decl.decls);
            try w.writeAll(" fields={");
            if (struct_decl.fields.len > 0) {
                try w.writeByte('\n');
                var it = struct_decl.fields.iterate(zir);
                while (it.next()) |field| {
                    try w.writeByteNTimes(' ', indent + 2);
                    if (field.is_comptime) try w.writeAll("comptime ");
                    try w.print("{} type=", .{std.zig.fmtId(field.name.get(zir))});
                    try dumpBody(w, indent + 2, zir, field.type_body);
                    if (field.align_body) |align_body| {
                        try w.writeAll(" align=");
                        try dumpBody(w, indent + 2, zir, align_body);
                    }
                    if (field.init_body) |init_body| {
                        try w.writeAll(" init=");
                        try dumpBody(w, indent + 2, zir, init_body);
                    }
                    try w.writeByte('\n');
                }
                try w.writeByteNTimes(' ', indent);
            }
            try w.writeByte('}');
        },

        .int_type => |int| try w.print("int_type({s}, {d})", .{ @tagName(int.signedness), int.bits }),

        .declaration => unreachable,
    }
    try w.writeAll("\n");
}

fn printDecl(w: anytype, indent: u32, zir: *const Zir, decl_inst: Zir.Inst.Index) (@TypeOf(w).Error)!void {
    const decl = decl_inst.get(zir).declaration;
    try w.writeByteNTimes(' ', indent);
    try w.print("%{d} = declaration(", .{@intFromEnum(decl_inst)});
    if (decl.is_pub) try w.writeAll("pub ");
    switch (decl.linkage) {
        .normal => {},
        .@"export" => try w.writeAll("export "),
        .@"extern" => try w.writeAll("extern "),
    }
    if (decl.lib_name != .empty) try w.print(" \"{}\"", .{std.zig.fmtEscapes(decl.lib_name.get(zir))});
    if (decl.is_threadlocal) try w.writeAll("threadlocal ");
    try w.writeAll(@tagName(decl.kind));
    if (decl.name != .empty) try w.print(" {}", .{std.zig.fmtId(decl.name.get(zir))});
    if (decl.value_body) |b| {
        try w.writeAll(" value_body=");
        try dumpBody(w, indent, zir, b);
    }
    if (decl.type_body) |b| {
        try w.writeAll(" type_body=");
        try dumpBody(w, indent, zir, b);
    }
    if (decl.align_body) |b| {
        try w.writeAll(" align_body=");
        try dumpBody(w, indent, zir, b);
    }
    if (decl.linksection_body) |b| {
        try w.writeAll(" linksection_body=");
        try dumpBody(w, indent, zir, b);
    }
    if (decl.addrspace_body) |b| {
        try w.writeAll(" addrspace_body=");
        try dumpBody(w, indent, zir, b);
    }
    if (decl.fn_info) |fn_info| {
        if (fn_info.is_inferred_error) try w.writeAll(" ies");
        if (fn_info.is_var_args) try w.writeAll(" varargs");
        if (fn_info.is_noinline) try w.writeAll(" noinline");
        switch (fn_info.@"callconv") {
            .auto => {},
            .@"inline" => try w.writeAll(" callconv=inline"),
            .body => |b| {
                try w.writeAll(" callconv=body=");
                try dumpBody(w, indent, zir, b);
            },
        }
        if (fn_info.params.len == 0) {
            try w.writeAll(" params={}");
        } else {
            try w.writeAll(" params={\n");
            var it = fn_info.params.iterate(zir);
            while (it.next()) |param| {
                try w.writeByteNTimes(' ', indent + 2);
                try w.print("%{d} = ", .{@intFromEnum(param.placeholder_inst)});
                if (param.is_comptime) try w.writeAll("comptime ");
                if (param.is_noalias) try w.writeAll("noalias ");
                try w.print("{}", .{std.zig.fmtId(param.name.get(zir))});
                if (param.type_body) |b| {
                    try w.writeAll(" type_body=");
                    try dumpBody(w, indent + 2, zir, b);
                } else {
                    try w.writeAll(" anytype");
                }
                if (param.type_is_generic) try w.writeAll(" type_generic");
                try w.writeAll(",\n");
            }
            try w.writeByteNTimes(' ', indent);
            try w.writeByte('}');
        }
        try w.writeAll(" ret_ty_body=");
        try dumpBody(w, indent, zir, fn_info.ret_ty_body);
        if (fn_info.ret_ty_is_generic) try w.writeAll(" ret_ty_generic");
        if (fn_info.body) |b| {
            try w.writeAll(" body=");
            try dumpBody(w, indent, zir, b);
        }
    }
    try w.writeAll(")");
}

fn printDecls(w: anytype, indent: u32, zir: *const Zir, decls: Zir.Declarations) (@TypeOf(w).Error)!void {
    if (decls.len == 0) return w.writeAll("{}");
    try w.writeAll("{\n");
    var it = decls.iterate(zir);
    while (it.next()) |decl_inst| {
        try printDecl(w, indent + 2, zir, decl_inst);
        try w.writeByte('\n');
    }
    try w.writeByteNTimes(' ', indent);
    try w.writeByte('}');
}

fn dumpBody(w: anytype, indent: u32, zir: *const Zir, body: Zir.Body) (@TypeOf(w).Error)!void {
    try w.writeAll("{\n");
    var it = body.iterate(zir);
    while (it.next()) |body_inst| {
        try dumpInner(w, indent + 2, zir, body_inst);
    }
    try w.writeByteNTimes(' ', indent);
    try w.writeByte('}');
}

fn fmtRef(ref: Zir.Ref) struct {
    ref: Zir.Ref,
    pub fn format(ctx: @This(), comptime _: []const u8, _: std.fmt.FormatOptions, w: anytype) !void {
        if (ctx.ref.toIndex()) |idx| {
            return w.print("%{d}", .{@intFromEnum(idx)});
        } else {
            return w.print("@{s}", .{@tagName(ctx.ref)});
        }
    }
} {
    return .{ .ref = ref };
}

const std = @import("std");
const Zir = @import("Zir.zig");
