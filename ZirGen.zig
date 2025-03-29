gpa: Allocator,
tree: *const Ast,
annotations: *const AstAnnotate.Result,

wip_instructions: std.MultiArrayList(Zir.Inst.Repr),
wip_extra: std.ArrayListUnmanaged(u32),
wip_limbs: std.ArrayListUnmanaged(std.math.big.Limb),
wip_string_bytes: std.ArrayListUnmanaged(u8),

wip_compile_errors: std.ArrayListUnmanaged(Zir.CompileError),
wip_error_notes: std.ArrayListUnmanaged(Zir.CompileError.Note),

wip_imports: std.AutoArrayHashMapUnmanaged(Zir.NullTerminatedString, void),

string_table: std.HashMapUnmanaged(u32, void, std.hash_map.StringIndexContext, std.hash_map.default_max_load_percentage),

scratch: std.ArrayListUnmanaged(u32),

scopes: std.ArrayListUnmanaged(Scope),

const Error = error{
    OutOfMemory,
    AnalysisFail,
};

const ScopeIterator = struct {
    scopes: []Scope,
    prev_idx: usize,
    fn next(si: *ScopeIterator) ?*Scope {
        if (si.prev_idx == 0) return null;
        si.prev_idx -= 1;
        return &si.scopes[si.prev_idx];
    }
};
fn iterateScopes(zg: *ZirGen) ScopeIterator {
    return .{
        .scopes = zg.scopes.items,
        .prev_idx = zg.scopes.items.len,
    };
}

fn addInst(zg: *ZirGen, tag: Zir.Inst.Repr.Tag, data: [2]u32) Allocator.Error!Zir.Inst.Index {
    const idx: Zir.Inst.Index = @enumFromInt(zg.wip_instructions.len);
    try zg.wip_instructions.append(zg.gpa, .{ .tag = tag, .data = data });
    return idx;
}
fn addExtended(zg: *ZirGen, tag: Zir.Inst.Repr.Extended.Tag, small: u16, operand: u32) Allocator.Error!Zir.Inst.Index {
    const idx: Zir.Inst.Index = @enumFromInt(zg.wip_instructions.len);
    try zg.wip_instructions.append(zg.gpa, .{
        .tag = .extended,
        .data = .{
            @bitCast(@as(Zir.Inst.Repr.Extended, .{
                .tag = tag,
                .small = small,
            })),
            operand,
        },
    });
    return idx;
}
fn coerce(zg: *ZirGen, operand: Zir.Ref, dest_ty: Zir.Ref) Allocator.Error!Zir.Ref {
    // Some coercions can be elided, because we know the type of `operand` already matches `dest_ty`.
    // e.g. `Ref.one` coerced to `Ref.comptime_int_type`, or `Ref.u0_type` coerced to `Ref.type_type`.
    // Other coercions can be done without any instructions, because there is a corresponding `Ref`.
    // e.g. `Ref.one` coerced to `Ref.usize_type` is `Ref.one_usize`.
    const CoercePair = packed struct(u64) {
        ty: Zir.Ref,
        val: Zir.Ref,

        // TODO: this will be unnecessary after https://github.com/ziglang/zig/issues/22214.
        // Until then, we use this, because `@as(u64, @bitCast(@as(CoercePair, .{ ... })))` is quite a mouthful.
        // Post-#22214, the `switch` cases will just be `.{ ... }`.
        fn init(ty: Zir.Ref, val: Zir.Ref) u64 {
            return @bitCast(@as(@This(), .{
                .ty = ty,
                .val = val,
            }));
        }
    };
    const pair = CoercePair.init;
    switch (pair(dest_ty, operand)) {
        // nop type coercions
        pair(.type_type, .u0_type),
        pair(.type_type, .i0_type),
        pair(.type_type, .u1_type),
        pair(.type_type, .u8_type),
        pair(.type_type, .i8_type),
        pair(.type_type, .u16_type),
        pair(.type_type, .i16_type),
        pair(.type_type, .u29_type),
        pair(.type_type, .u32_type),
        pair(.type_type, .i32_type),
        pair(.type_type, .u64_type),
        pair(.type_type, .i64_type),
        pair(.type_type, .u80_type),
        pair(.type_type, .u128_type),
        pair(.type_type, .i128_type),
        pair(.type_type, .usize_type),
        pair(.type_type, .isize_type),
        pair(.type_type, .c_char_type),
        pair(.type_type, .c_short_type),
        pair(.type_type, .c_ushort_type),
        pair(.type_type, .c_int_type),
        pair(.type_type, .c_uint_type),
        pair(.type_type, .c_long_type),
        pair(.type_type, .c_ulong_type),
        pair(.type_type, .c_longlong_type),
        pair(.type_type, .c_ulonglong_type),
        pair(.type_type, .c_longdouble_type),
        pair(.type_type, .f16_type),
        pair(.type_type, .f32_type),
        pair(.type_type, .f64_type),
        pair(.type_type, .f80_type),
        pair(.type_type, .f128_type),
        pair(.type_type, .anyopaque_type),
        pair(.type_type, .bool_type),
        pair(.type_type, .void_type),
        pair(.type_type, .type_type),
        pair(.type_type, .anyerror_type),
        pair(.type_type, .comptime_int_type),
        pair(.type_type, .comptime_float_type),
        pair(.type_type, .noreturn_type),
        pair(.type_type, .anyframe_type),
        pair(.type_type, .null_type),
        pair(.type_type, .undefined_type),
        pair(.type_type, .enum_literal_type),
        pair(.type_type, .manyptr_u8_type),
        pair(.type_type, .manyptr_const_u8_type),
        pair(.type_type, .manyptr_const_u8_sentinel_0_type),
        pair(.type_type, .single_const_pointer_to_comptime_int_type),
        pair(.type_type, .slice_const_u8_type),
        pair(.type_type, .slice_const_u8_sentinel_0_type),
        pair(.type_type, .vector_16_i8_type),
        pair(.type_type, .vector_32_i8_type),
        pair(.type_type, .vector_16_u8_type),
        pair(.type_type, .vector_32_u8_type),
        pair(.type_type, .vector_8_i16_type),
        pair(.type_type, .vector_16_i16_type),
        pair(.type_type, .vector_8_u16_type),
        pair(.type_type, .vector_16_u16_type),
        pair(.type_type, .vector_4_i32_type),
        pair(.type_type, .vector_8_i32_type),
        pair(.type_type, .vector_4_u32_type),
        pair(.type_type, .vector_8_u32_type),
        pair(.type_type, .vector_2_i64_type),
        pair(.type_type, .vector_4_i64_type),
        pair(.type_type, .vector_2_u64_type),
        pair(.type_type, .vector_4_u64_type),
        pair(.type_type, .vector_4_f16_type),
        pair(.type_type, .vector_8_f16_type),
        pair(.type_type, .vector_2_f32_type),
        pair(.type_type, .vector_4_f32_type),
        pair(.type_type, .vector_8_f32_type),
        pair(.type_type, .vector_2_f64_type),
        pair(.type_type, .vector_4_f64_type),
        pair(.type_type, .optional_noreturn_type),
        pair(.type_type, .anyerror_void_error_union_type),
        pair(.type_type, .adhoc_inferred_error_set_type),
        pair(.type_type, .generic_poison_type),
        pair(.type_type, .empty_tuple_type),
        // other nop coercions
        pair(.undefined_type, .undef),
        pair(.comptime_int_type, .zero),
        pair(.comptime_int_type, .one),
        pair(.comptime_int_type, .negative_one),
        pair(.usize_type, .zero_usize),
        pair(.usize_type, .one_usize),
        pair(.u8_type, .zero_u8),
        pair(.u8_type, .one_u8),
        pair(.u8_type, .four_u8),
        pair(.void_type, .void_value),
        pair(.noreturn_type, .unreachable_value),
        pair(.null_type, .null_value),
        pair(.bool_type, .bool_true),
        pair(.bool_type, .bool_false),
        pair(.empty_tuple_type, .empty_tuple),
        => return operand,

        // Coercions between constant `Ref`s
        pair(.comptime_int_type, .zero_usize) => return .zero,
        pair(.comptime_int_type, .zero_u8) => return .zero,
        pair(.comptime_int_type, .one_usize) => return .one,
        pair(.comptime_int_type, .one_u8) => return .one,
        pair(.u8_type, .zero) => return .zero_u8,
        pair(.u8_type, .zero_usize) => return .zero_u8,
        pair(.u8_type, .one) => return .one_u8,
        pair(.u8_type, .one_usize) => return .one_u8,
        pair(.usize_type, .zero) => return .zero_usize,
        pair(.usize_type, .zero_u8) => return .zero_usize,
        pair(.usize_type, .one) => return .one_usize,
        pair(.usize_type, .one_u8) => return .one_usize,

        // We have to add the coercion instruction
        else => {},
    }
    return (try zg.addInst(.coerce, .{
        @intFromEnum(operand),
        @intFromEnum(dest_ty),
    })).toRef();
}
fn setData(zg: *ZirGen, inst: Zir.Inst.Index, data: [2]u32) void {
    zg.wip_instructions.items(.data)[@intFromEnum(inst)] = data;
}
fn setExtended(
    zg: *ZirGen,
    inst: Zir.Inst.Index,
    tag: Zir.Inst.Repr.Extended.Tag,
    small: u16,
    data1: u32,
) void {
    zg.setData(inst, .{
        @bitCast(@as(Zir.Inst.Repr.Extended, .{
            .tag = tag,
            .small = small,
        })),
        data1,
    });
}

fn beginExtra(zg: *ZirGen, total_len: u32) Allocator.Error!ExtraWriter {
    try zg.wip_extra.ensureUnusedCapacity(zg.gpa, total_len);
    return .{
        .zg = zg,
        .idx = @enumFromInt(zg.wip_extra.items.len),
        .prev_extra_len = if (std.debug.runtime_safety) @intCast(zg.wip_extra.items.len),
        .remaining_len = if (std.debug.runtime_safety) total_len,
    };
}
const ExtraWriter = struct {
    zg: *ZirGen,
    idx: Zir.ExtraIndex,
    prev_extra_len: if (std.debug.runtime_safety) u32 else void,
    remaining_len: if (std.debug.runtime_safety) u32 else void,
    fn append(ew: *ExtraWriter, elem: anytype) void {
        const zg = ew.zg;
        if (std.debug.runtime_safety) {
            assert(zg.wip_extra.items.len == ew.prev_extra_len);
            ew.remaining_len -= 1;
            ew.prev_extra_len += 1;
        }
        zg.wip_extra.appendAssumeCapacity(extraToU32(elem));
    }
    fn appendStruct(ew: *ExtraWriter, comptime T: type, x: T) void {
        var vals: [@typeInfo(T).@"struct".fields.len]u32 = undefined;
        inline for (&vals, @typeInfo(T).@"struct".fields) |*elem, field| {
            elem.* = extraToU32(@field(x, field.name));
        }
        ew.appendSlice(u32, &vals);
    }
    fn appendSlice(ew: *ExtraWriter, comptime T: type, x: []const T) void {
        const zg = ew.zg;
        if (std.debug.runtime_safety) {
            assert(zg.wip_extra.items.len == ew.prev_extra_len);
            ew.remaining_len -= @intCast(x.len);
            ew.prev_extra_len += @intCast(x.len);
        }
        _ = comptime u32ToExtra(T, 0); // check this is okay
        zg.wip_extra.appendSliceAssumeCapacity(@ptrCast(x));
    }
    fn finish(ew: *ExtraWriter) Zir.ExtraIndex {
        if (std.debug.runtime_safety) {
            assert(ew.zg.wip_extra.items.len == ew.prev_extra_len);
            assert(ew.remaining_len == 0);
        }
        const idx = ew.idx;
        ew.* = undefined;
        return idx;
    }
};
const WipFlagBits = struct {
    next_flag_idx: u32,
    cur_bag: u32,
    ew: *ExtraWriter,
    fn init(ew: *ExtraWriter) WipFlagBits {
        return .{
            .next_flag_idx = 0,
            .cur_bag = 0,
            .ew = ew,
        };
    }
    fn next(wip: *WipFlagBits, val: bool) void {
        const bit_idx: u5 = @intCast(wip.next_flag_idx % 32);
        if (val) wip.cur_bag |= (@as(u32, 1) << bit_idx);
        if (bit_idx == 31) {
            wip.ew.append(wip.cur_bag);
            wip.cur_bag = 0;
        }
        wip.next_flag_idx += 1;
    }
    fn finish(wip: *WipFlagBits) void {
        if (wip.next_flag_idx % 32 != 0) wip.ew.append(wip.cur_bag);
        wip.* = undefined;
    }
};

fn Scratch(comptime S: type) type {
    return struct {
        const Self = @This();

        comptime {
            for (@typeInfo(S).@"struct".fields) |f| {
                assert(@sizeOf(f.type) == 4);
            }
        }

        const scratch_per_item = @typeInfo(S).@"struct".fields.len;

        zg: *ZirGen,
        idx: usize,
        len: usize,

        fn empty(zg: *ZirGen) Self {
            return .{
                .zg = zg,
                .idx = zg.scratch.items.len,
                .len = 0,
            };
        }
        fn append(s: *Self, val: S) Allocator.Error!void {
            const zg = s.zg;
            assert(zg.scratch.items.len == s.idx + s.len * scratch_per_item);
            s.len += 1;
            const dest = try zg.scratch.addManyAsArray(zg.gpa, scratch_per_item);
            inline for (dest, @typeInfo(S).@"struct".fields) |*elem, f| {
                elem.* = extraToU32(@field(val, f.name));
            }
        }
        fn get(s: *Self, idx: usize) S {
            assert(idx < s.len);
            const scratch_idx = s.idx + idx * scratch_per_item;
            const elems = s.zg.scratch.items[scratch_idx..][0..scratch_per_item].*;
            var res: S = undefined;
            inline for (elems, @typeInfo(S).@"struct".fields) |elem, f| {
                @field(res, f.name) = u32ToExtra(f.type, elem);
            }
            return res;
        }
        fn free(s: *Self) void {
            const elems = scratch_per_item * s.len;
            assert(s.zg.scratch.items.len == s.idx + elems);
            s.zg.scratch.items.len = s.idx;
        }
    };
}

pub fn generate(gpa: Allocator, tree: *const Ast) Allocator.Error!Zir {
    var annotations = try AstAnnotate.annotate(gpa, tree);
    defer annotations.deinit(gpa);

    var zg: ZirGen = .{
        .gpa = gpa,
        .tree = tree,
        .annotations = &annotations,
        .wip_instructions = .empty,
        .wip_extra = .empty,
        .wip_limbs = .empty,
        .wip_string_bytes = .empty,
        .wip_compile_errors = .empty,
        .wip_error_notes = .empty,
        .wip_imports = .empty,
        .string_table = .empty,
        .scratch = .empty,
        .scopes = .empty,
    };
    defer {
        zg.wip_instructions.deinit(gpa);
        zg.wip_extra.deinit(gpa);
        zg.wip_limbs.deinit(gpa);
        zg.wip_string_bytes.deinit(gpa);
        zg.wip_compile_errors.deinit(gpa);
        zg.wip_error_notes.deinit(gpa);
        zg.wip_imports.deinit(gpa);
        zg.string_table.deinit(gpa);
        zg.scratch.deinit(gpa);
        zg.scopes.deinit(gpa);
    }

    // Before we do anything, we need to add "" to the string table.
    // `NullTerminatedString.empty` relies on this!
    try zg.wip_string_bytes.append(gpa, 0); // null terminator
    try zg.string_table.putNoClobberContext(gpa, 0, {}, std.hash_map.StringIndexContext{
        .bytes = &zg.wip_string_bytes,
    });

    const fatal = fatal: {
        if (tree.errors.len != 0) {
            try zg.lowerAstErrors();
            break :fatal true;
        }

        const root_ref = zg.structDeclInner(.root, tree.containerDeclRoot(), .auto, .none, .parent) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.AnalysisFail => break :fatal true,
        };

        assert(root_ref.toIndex().? == .root);
        break :fatal false;
    };

    var zir: Zir = .{
        .instructions = .empty,
        .extra = &.{},
        .limbs = &.{},
        .string_bytes = &.{},
        .compile_errors = &.{},
        .error_notes = &.{},
        .imports = &.{},
    };
    errdefer zir.deinit(gpa);

    // `fatal` is indicated by empty `zir.instructions`. Everything else is set regardless of `fatal`.
    if (!fatal) zir.instructions = zg.wip_instructions.toOwnedSlice();
    zir.extra = try zg.wip_extra.toOwnedSlice(gpa);
    zir.limbs = try zg.wip_limbs.toOwnedSlice(gpa);
    zir.string_bytes = try zg.wip_string_bytes.toOwnedSlice(gpa);
    zir.compile_errors = try zg.wip_compile_errors.toOwnedSlice(gpa);
    zir.error_notes = try zg.wip_error_notes.toOwnedSlice(gpa);
    // For `imports`, we were deduping with a hashmap, and now want to pull out just the keys.
    zir.imports = try gpa.dupe(Zir.NullTerminatedString, zg.wip_imports.keys());

    return zir;
}

const ExprResult = union(enum) {
    /// This expression diverts control flow; it is considered "always-noreturn".
    /// If there is a result location, it has not been written to.
    /// Node is the AST node which diverted control flow.
    @"unreachable": Ast.Node.Index,
    /// The expression may not divert control flow; it is not considered "always-noreturn".
    /// `Zir.Ref` is the result of the expression; its exact usage is defined by `ri`.
    reachable: Zir.Ref,
};

/// Emit ZIR to evaluate the expression at `node`. The instructions are added to the end of `zg.wip_instructions`.
/// If this expression is always-noreturn, `.@"unreachable"` is returned. Typically (e.g. when evaluating sub-expressions),
/// this should be a compile error; see `reachableExpr` for a different function which emits an error when the expression
/// is always-noreturn.
fn expr(
    zg: *ZirGen,
    ri: ResultInfo,
    node: Ast.Node.Index,
) Error!ExprResult {
    const tree = zg.tree;
    switch (tree.nodeTag(node)) {
        // Container-level nodes
        .root,
        .@"usingnamespace",
        .test_decl,
        .container_field_init,
        .container_field_align,
        .container_field,
        .fn_decl,
        => unreachable,

        // Statements, not expressions; see `blockStatements`
        .global_var_decl,
        .local_var_decl,
        .simple_var_decl,
        .aligned_var_decl,
        .@"defer",
        .@"errdefer",
        => unreachable,

        // Other non-expression nodes
        .switch_case,
        .switch_case_inline,
        .switch_case_one,
        .switch_case_inline_one,
        .switch_range,
        .asm_output,
        .asm_input,
        .for_range,
        => unreachable,

        else => @panic("TODO"),

        .@"return" => return zg.retExpr(node),

        .assign => return zg.assignExpr(ri, node),

        .grouped_expression => return zg.expr(ri, tree.nodeData(node).node_and_token[0]),
        .number_literal => return zg.numberLiteral(ri, node, .positive),
        .string_literal => return zg.stringLiteral(ri, node),

        .negation => return zg.negationExpr(ri, node),

        // zig fmt: off
        .add      => return zg.simpleBinOpExpr(ri, node, .add),
        .add_wrap => return zg.simpleBinOpExpr(ri, node, .add_wrap),
        .add_sat  => return zg.simpleBinOpExpr(ri, node, .add_sat),
        .sub      => return zg.simpleBinOpExpr(ri, node, .sub),
        .sub_wrap => return zg.simpleBinOpExpr(ri, node, .sub_wrap),
        .sub_sat  => return zg.simpleBinOpExpr(ri, node, .sub_sat),
        .mul      => return zg.simpleBinOpExpr(ri, node, .mul),
        .mul_wrap => return zg.simpleBinOpExpr(ri, node, .mul_wrap),
        .mul_sat  => return zg.simpleBinOpExpr(ri, node, .mul_sat),
        .div      => return zg.simpleBinOpExpr(ri, node, .div),
        .mod      => return zg.simpleBinOpExpr(ri, node, .mod_rem),
        .shl_sat  => return zg.simpleBinOpExpr(ri, node, .shl_sat),
        // zig fmt: on

        .block_two,
        .block_two_semicolon,
        .block,
        .block_semicolon,
        => return zg.blockExpr(ri, node),

        .@"break" => return zg.breakExpr(node),

        .identifier => return zg.identifierExpr(ri, node),

        .address_of => return zg.expr(.{
            .is_direct_discard = false,
            .eval_mode = .lvalue_mutable,
            .ty = ri.ty,
            .loc = ri.loc,
        }, tree.nodeData(node).node),

        .builtin_call_two,
        .builtin_call_two_comma,
        .builtin_call,
        .builtin_call_comma,
        => return zg.builtinCallExpr(ri, node),
    }
}

fn reachableExpr(
    zg: *ZirGen,
    ri: ResultInfo,
    node: Ast.Node.Index,
    /// If `node` is always-noreturn, then we use this as the "unreachable code" source location.
    reachable_node: Ast.Node.Index,
) Error!Zir.Ref {
    switch (try zg.expr(ri, node)) {
        .reachable => |ref| return ref,
        .@"unreachable" => |noreturn_node| {
            try zg.addUnreachableCodeError(noreturn_node, reachable_node);
            return .void_value;
        },
    }
}
fn addUnreachableCodeError(
    zg: *ZirGen,
    noreturn_node: Ast.Node.Index,
    unreachable_node: Ast.Node.Index,
) Allocator.Error!void {
    try zg.addError(.{ .node = unreachable_node }, "unreachable code", .{}, &.{
        try zg.errNote(.{ .node = noreturn_node }, "control flow diverted here", .{}),
    });
}
fn blockStatements(
    zg: *ZirGen,
    stmts: []const Ast.Node.Index,
) Error!union(enum) {
    /// This body diverts control flow; the latest instruction is noreturn.
    /// Node is the AST node which diverted control flow.
    @"unreachable": Ast.Node.Index,
    /// This body does not (necessarily) divert control flow.
    /// In other words, the end of the body is potentially reachable.
    reachable,
} {
    const tree = zg.tree;
    const gpa = zg.gpa;

    const old_scopes_len = zg.scopes.items.len;
    defer zg.scopes.shrinkRetainingCapacity(old_scopes_len);

    for (stmts, 0..) |stmt_node, stmt_idx| {
        switch (tree.nodeTag(stmt_node)) {
            .global_var_decl,
            .local_var_decl,
            .simple_var_decl,
            .aligned_var_decl,
            => try zg.scopes.append(gpa, try zg.varDecl(stmt_node)),

            else => switch (try zg.expr(.simple_rvalue, stmt_node)) {
                .reachable => |result| {
                    try zg.ensureResultUsed(result);
                },
                .@"unreachable" => |noreturn_node| {
                    if (stmt_idx != stmts.len - 1) {
                        try zg.addUnreachableCodeError(noreturn_node, stmts[stmt_idx + 1]);
                    } else {
                        try zg.checkUsed(old_scopes_len);
                    }
                    return .{ .@"unreachable" = noreturn_node };
                },
            },
        }
    }
    try zg.checkUsed(old_scopes_len);
    return .reachable;
}

fn checkUsed(zg: *ZirGen, scopes_start: usize) Allocator.Error!void {
    for (zg.scopes.items[scopes_start..]) |scope| switch (scope) {
        .local_val => |lv| {
            if (lv.used == .none and lv.discarded == .none) {
                try zg.addError(.token(lv.name_token), "unused {s}", .{@tagName(lv.id_cat)}, &.{
                    try zg.errNote(.token(lv.name_token), "suppress this error by explicitly discarding the {s} with '_ = {}'", .{
                        @tagName(lv.id_cat),
                        std.zig.fmtId(zg.getString(lv.name)),
                    }),
                });
            } else {
                if (lv.used.unwrap()) |use_tok| {
                    if (lv.discarded.unwrap()) |discard_tok| {
                        try zg.addError(.token(discard_tok), "pointless discard of {s}", .{@tagName(lv.id_cat)}, &.{
                            try zg.errNote(.token(use_tok), "used here", .{}),
                        });
                    }
                }
            }
        },

        .namespace,
        .break_target,
        => unreachable, // should be popped already
    };
}

fn varDecl(zg: *ZirGen, node: Ast.Node.Index) Error!Scope {
    const tree = zg.tree;

    const full = tree.fullVarDecl(node).?;

    const is_const = switch (tree.tokenTag(full.ast.mut_token)) {
        .keyword_const => true,
        .keyword_var => false,
        else => unreachable,
    };

    const name_token = full.ast.mut_token + 1;
    if (mem.eql(u8, tree.tokenSlice(name_token), "_")) {
        return zg.fail(.token(name_token), "'_' used as an identifier without @\"_\" syntax", .{}, &.{});
    }
    const ident = try zg.identAsString(name_token);

    // TODO detect shadowing

    const init_node = full.ast.init_node.unwrap() orelse {
        return zg.fail(.{ .node = node }, "variables must be initialized", .{}, &.{});
    };

    if (full.ast.addrspace_node.unwrap()) |addrspace_node| return zg.fail(
        .{ .node = addrspace_node },
        "cannot set address space of local variable '{}'",
        .{std.zig.fmtId(zg.getString(ident))},
        &.{},
    );
    if (full.ast.section_node.unwrap()) |section_node| return zg.fail(
        .{ .node = section_node },
        "cannot set section of local variable '{}'",
        .{std.zig.fmtId(zg.getString(ident))},
        &.{},
    );

    if (full.ast.align_node.unwrap()) |_| @panic("TODO alloc impl");

    if (full.ast.type_node.unwrap()) |_| @panic("TODO result types");

    if (is_const) {
        if (full.comptime_token) |comptime_token| {
            try zg.addError(.token(comptime_token), "'comptime const' is redundant; wrap the initialization expression with 'comptime' instead", .{}, &.{});
        }

        // `comptime const` is a non-fatal error; treat it like the init was marked `comptime`.
        const force_comptime = full.comptime_token != null;

        _ = force_comptime; // TODO

        const need_ref_inst = zg.annotations.consts_need_ref.contains(name_token);

        // TODO: result location
        const init_ref = try zg.reachableExpr(.simple_rvalue, init_node, node);
        const validate_inst = try zg.addExtended(if (need_ref_inst) .validate_const_ref else .validate_const, 0, @intFromEnum(init_ref));
        return .{ .local_val = .{
            .name = ident,
            .val = init_ref,
            .ptr_strat = if (need_ref_inst) .{ .ref = validate_inst.toRef() } else .unused,
            .name_token = name_token,
            .id_cat = .@"local constant",
        } };
    } else {
        @panic("TODO alloc impl");
    }
}

fn numberLiteral(zg: *ZirGen, ri: ResultInfo, node: Ast.Node.Index, sign: enum { negative, positive }) Error!ExprResult {
    const tree = zg.tree;
    const token = tree.nodeMainToken(node);
    const bytes = tree.tokenSlice(token);
    const result: Zir.Ref = switch (std.zig.parseNumberLiteral(bytes)) {
        .int => |num| switch (num) {
            0 => switch (sign) {
                .positive => .zero,
                .negative => return zg.fail(.{ .node = node }, "integer literal '-0' is ambiguous", .{}, &.{
                    try zg.errNote(.{ .node = node }, "use '0' for an integer zero", .{}),
                    try zg.errNote(.{ .node = node }, "use '-0.0' for a floating-point signed zero", .{}),
                }),
            },
            1 => switch (sign) {
                .positive => .one,
                .negative => .negative_one,
            },
            else => res: {
                const pos = (try zg.addInst(.int, @bitCast(num))).toRef();
                switch (sign) {
                    .positive => break :res pos,
                    .negative => break :res (try zg.addExtended(.negate, 0, @intFromEnum(pos))).toRef(),
                }
            },
        },
        .big_int => |base| res: {
            const gpa = zg.gpa;
            var big: std.math.big.int.Managed = try .init(gpa);
            defer big.deinit();
            const prefix_offset: usize = if (base == .decimal) 0 else 2;
            big.setString(@intFromEnum(base), bytes[prefix_offset..]) catch |err| switch (err) {
                error.InvalidCharacter => unreachable, // caught in `parseNumberLiteral`
                error.InvalidBase => unreachable, // we only pass 16, 8, 2, see above
                error.OutOfMemory => return error.OutOfMemory,
            };
            assert(big.isPositive());
            const limbs = big.limbs[0..big.len()];
            const limbs_len = std.math.cast(u16, limbs.len) orelse {
                // Assuming 32-bit limbs, 65535 limbs is ~2.1m bits, which is ~630k decimal bytes.
                // Big enough that we clearly don't need to support it, but small enough that we
                // do actually need to have an error message!
                return zg.fail(.{ .node = node }, "this compiler implementation does not support integer literals this large", .{}, &.{});
            };
            const limbs_idx: u32 = @intCast(zg.wip_limbs.items.len);
            try zg.wip_limbs.appendSlice(gpa, limbs);
            const pos = (try zg.addExtended(.int_big, limbs_len, limbs_idx)).toRef();
            switch (sign) {
                .positive => break :res pos,
                .negative => break :res (try zg.addExtended(.negate, 0, @intFromEnum(pos))).toRef(),
            }
        },
        .float => res: {
            const unsigned_float = std.fmt.parseFloat(f128, bytes) catch |err| switch (err) {
                error.InvalidCharacter => unreachable, // validated by tokenizer
            };
            const float = switch (sign) {
                .negative => -unsigned_float,
                .positive => unsigned_float,
            };
            // If the value fits into an f64 without losing any precision, store it that way.
            const smaller_float: f64 = @floatCast(float);
            const bigger_again: f128 = smaller_float;
            if (bigger_again == float) {
                break :res (try zg.addInst(.float64, @bitCast(smaller_float))).toRef();
            }
            // We need to use 128 bits. Break the float into 4 u32 values so we can put it in `extra`.
            var extra = try zg.beginExtra(4);
            extra.appendSlice(u32, @ptrCast((&float)[0..1]));
            break :res (try zg.addExtended(.float128, 0, @intFromEnum(extra.finish()))).toRef();
        },
        .failure => |err| return zg.failWithNumberError(err, token, bytes),
    };
    return .{ .reachable = try zg.rvalue(ri, result) };
}

fn stringLiteral(zg: *ZirGen, ri: ResultInfo, node: Ast.Node.Index) Error!ExprResult {
    const str = try zg.strLitAsString(zg.tree.nodeMainToken(node));
    const inst = try zg.addInst(.str, .{
        str.index,
        str.len,
    });
    return .{ .reachable = try zg.rvalue(ri, inst.toRef()) };
}

fn identifierExpr(zg: *ZirGen, ri: ResultInfo, node: Ast.Node.Index) Error!ExprResult {
    const tree = zg.tree;
    const token = tree.nodeMainToken(node);
    const ident_raw = tree.tokenSlice(token);

    if (mem.eql(u8, ident_raw, "_")) {
        return zg.fail(.{ .node = node }, "'_' used as an identifier without @\"_\" syntax", .{}, &.{});
    }

    if (ident_raw[0] != '@') {
        if (primitive_instrs.get(ident_raw)) |result| {
            return .{ .reachable = try zg.rvalue(ri, result) };
        }
        int_ty: {
            const signedness: std.builtin.Signedness = switch (ident_raw[0]) {
                'u' => .unsigned,
                'i' => .signed,
                else => break :int_ty,
            };
            const bit_count = parseBitCount(ident_raw[1..]) catch |err| switch (err) {
                error.Overflow => return zg.fail(
                    .{ .node = node },
                    "primitive integer type '{s}' exceeds maximum bit width of 65535",
                    .{ident_raw},
                    &.{},
                ),
                error.InvalidCharacter => break :int_ty,
            };
            if (ident_raw.len >= 3 and ident_raw[1] == '0') return zg.fail(
                .{ .node = node },
                "primitive integer type '{s}' has leading zero",
                .{ident_raw},
                &.{},
            );
            const result = try zg.addExtended(.int_type, bit_count, @intFromEnum(signedness));
            return .{ .reachable = try zg.rvalue(ri, result.toRef()) };
        }
    }

    const ident_str = try zg.identAsString(token);

    var found_decl: ?Ast.Node.Index = null;
    var it = zg.iterateScopes();
    while (it.next()) |scope| {
        switch (scope.*) {
            .local_val => |*lv| {
                if (lv.name == ident_str) {
                    const maybe_ref: Zir.Ref = switch (ri.eval_mode) {
                        .rvalue => lv.val,
                        .lvalue_mutable, .lvalue_immutable => try lv.ptr(zg),
                    };
                    if (ri.is_direct_discard) {
                        if (lv.discarded == .none) lv.discarded = .fromToken(token);
                    } else {
                        if (lv.used == .none) lv.used = .fromToken(token);
                    }
                    if (lv.used_or_discarded_ptr) |p| p.* = true;
                    return .{ .reachable = try zg.applyResultTypeLocation(ri, maybe_ref) };
                }
            },
            .namespace => |ns| {
                if (ns.members.get(ident_str)) |decl_node| {
                    if (found_decl) |existing_decl_node| {
                        return zg.fail(.{ .node = node }, "ambiguous reference", .{}, &.{
                            try zg.errNote(.{ .node = existing_decl_node }, "declared here", .{}),
                            try zg.errNote(.{ .node = decl_node }, "also declared here", .{}),
                        });
                    } else {
                        found_decl = decl_node;
                    }
                }
            },
            else => {},
        }
    }
    if (found_decl != null) {
        // TODO: tunnel through dem closures boiiiii
        const ref = (try zg.addExtended(.decl_ref, 0, @intFromEnum(ident_str))).toRef();
        const maybe_loaded = switch (ri.eval_mode) {
            .rvalue => (try zg.addExtended(.load, 0, @intFromEnum(ref))).toRef(),
            .lvalue_mutable, .lvalue_immutable => ref,
        };
        return .{ .reachable = try zg.applyResultTypeLocation(ri, maybe_loaded) };
    } else {
        @panic("TODO error");
    }
}
fn parseBitCount(buf: []const u8) std.fmt.ParseIntError!u16 {
    if (buf.len == 0) return error.InvalidCharacter;
    var x: u16 = 0;
    for (buf) |c| {
        const digit = switch (c) {
            '0'...'9' => c - '0',
            else => return error.InvalidCharacter,
        };
        if (x != 0) x = try std.math.mul(u16, x, 10);
        x = try std.math.add(u16, x, digit);
    }
    return x;
}
const primitive_instrs: std.StaticStringMap(Zir.Ref) = .initComptime(.{
    .{ "anyerror", .anyerror_type },
    .{ "anyframe", .anyframe_type },
    .{ "anyopaque", .anyopaque_type },
    .{ "bool", .bool_type },
    .{ "c_int", .c_int_type },
    .{ "c_long", .c_long_type },
    .{ "c_longdouble", .c_longdouble_type },
    .{ "c_longlong", .c_longlong_type },
    .{ "c_char", .c_char_type },
    .{ "c_short", .c_short_type },
    .{ "c_uint", .c_uint_type },
    .{ "c_ulong", .c_ulong_type },
    .{ "c_ulonglong", .c_ulonglong_type },
    .{ "c_ushort", .c_ushort_type },
    .{ "comptime_float", .comptime_float_type },
    .{ "comptime_int", .comptime_int_type },
    .{ "f128", .f128_type },
    .{ "f16", .f16_type },
    .{ "f32", .f32_type },
    .{ "f64", .f64_type },
    .{ "f80", .f80_type },
    .{ "false", .bool_false },
    .{ "i16", .i16_type },
    .{ "i32", .i32_type },
    .{ "i64", .i64_type },
    .{ "i128", .i128_type },
    .{ "i8", .i8_type },
    .{ "isize", .isize_type },
    .{ "noreturn", .noreturn_type },
    .{ "null", .null_value },
    .{ "true", .bool_true },
    .{ "type", .type_type },
    .{ "u16", .u16_type },
    .{ "u29", .u29_type },
    .{ "u32", .u32_type },
    .{ "u64", .u64_type },
    .{ "u128", .u128_type },
    .{ "u1", .u1_type },
    .{ "u8", .u8_type },
    .{ "undefined", .undef },
    .{ "usize", .usize_type },
    .{ "void", .void_type },
});
comptime {
    // These checks ensure that std.zig.primitives stays in sync with the primitive->Zir map.
    for (primitive_instrs.keys(), primitive_instrs.values()) |key, value| {
        if (!std.zig.primitives.isPrimitive(key)) {
            @compileError("std.zig.isPrimitive() is not aware of Zir instr '" ++ @tagName(value) ++ "'");
        }
    }
    for (std.zig.primitives.names.keys()) |key| {
        if (primitive_instrs.get(key) == null) {
            @compileError("std.zig.primitives entry '" ++ key ++ "' does not have a corresponding Zir instr");
        }
    }
}

fn builtinCallExpr(zg: *ZirGen, ri: ResultInfo, node: Ast.Node.Index) Error!ExprResult {
    const tree = zg.tree;

    var buf: [2]Ast.Node.Index = undefined;
    const params = tree.builtinCallParams(&buf, node).?;

    const builtin_token = tree.nodeMainToken(node);
    const builtin_name = tree.tokenSlice(builtin_token);

    const info = BuiltinFn.list.get(builtin_name) orelse return zg.fail(
        .{ .node = node },
        "invalid builtin function: '{s}'",
        .{builtin_name},
        &.{},
    );
    if (info.param_count) |expected| {
        if (expected != params.len) {
            return zg.fail(.{ .node = node }, "expected {d} argument{s}, found {d}", .{
                expected,
                if (expected == 1) "" else "s",
                params.len,
            }, &.{});
        }
    }

    // TODO check `info.illegal_outside_function`

    switch (info.tag) {
        .import => {
            const operand_node = params[0];
            if (tree.nodeTag(operand_node) != .string_literal) {
                // Spec reference: https://github.com/ziglang/zig/issues/2206
                return zg.fail(.{ .node = operand_node }, "@import operand must be a string literal", .{}, &.{});
            }
            const str = try zg.restrictedStrLitAsString(tree.nodeMainToken(operand_node), "import path");
            const res_ty: Zir.Ref = .none; // TODO
            const result = try zg.addInst(.import, .{ @intFromEnum(res_ty), @intFromEnum(str) });
            try zg.wip_imports.put(zg.gpa, str, {});
            return .{ .reachable = try zg.rvalue(ri, result.toRef()) };
        },
        .as => {
            // TODO: typeExpr? is_comptime?
            const dest_type = try zg.reachableExpr(.simple_rvalue, params[0], node);
            const result = try zg.reachableExpr(.{
                .is_direct_discard = false,
                .eval_mode = .rvalue,
                .ty = .{ .coerce = dest_type },
                .loc = .none,
            }, params[1], node);
            return .{ .reachable = try zg.rvalue(ri, result) };
        },
        else => @panic("TODO"),
    }
}

fn retExpr(zg: *ZirGen, node: Ast.Node.Index) Error!ExprResult {
    const tree = zg.tree;
    // TODO: check in function
    // TODO: check we're not in a defer
    // TODO: hot path for "return error.Foo"
    // TODO: run defers
    // TODO: ret_ptr if nodes_need_rl
    // etc
    if (tree.nodeData(node).opt_node.unwrap()) |operand_node| {
        const operand = try zg.reachableExpr(.simple_rvalue, operand_node, node);
        _ = try zg.addExtended(.ret, 0, @intFromEnum(operand));
    } else {
        _ = try zg.addExtended(.ret, 0, @intFromEnum(Zir.Ref.void_value));
    }
    return .{ .@"unreachable" = node };
}

fn assignExpr(zg: *ZirGen, ri: ResultInfo, node: Ast.Node.Index) Error!ExprResult {
    const tree = zg.tree;
    const lhs_node, const rhs_node = tree.nodeData(node).node_and_node;

    const is_discard: bool = d: {
        if (tree.nodeTag(lhs_node) != .identifier) break :d false;
        // This intentionally does not support `@"_"` syntax.
        const ident_name = tree.tokenSlice(tree.nodeMainToken(lhs_node));
        break :d mem.eql(u8, ident_name, "_");
    };
    if (is_discard) {
        _ = try zg.reachableExpr(.{
            .is_direct_discard = true,
            .eval_mode = .rvalue,
            .ty = .none,
            .loc = .discard,
        }, rhs_node, node);
    } else {
        const dest_ptr = try zg.reachableExpr(.{
            .is_direct_discard = false,
            .eval_mode = .lvalue_mutable,
            .ty = .none,
            .loc = .none,
        }, lhs_node, node);
        _ = try zg.reachableExpr(.{
            .is_direct_discard = false,
            .eval_mode = .rvalue,
            .ty = .implicit_from_loc,
            .loc = .{ .ptr = dest_ptr },
        }, rhs_node, node);
    }

    return .{ .reachable = try zg.rvalue(ri, .void_value) };
}

fn negationExpr(
    zg: *ZirGen,
    ri: ResultInfo,
    node: Ast.Node.Index,
) Error!ExprResult {
    const tree = zg.tree;
    const operand_node = tree.nodeData(node).node;
    if (tree.nodeTag(operand_node) == .number_literal) {
        return zg.numberLiteral(ri, operand_node, .negative);
    }
    const operand = try zg.reachableExpr(.simple_rvalue, operand_node, node);
    const inst = try zg.addExtended(.negate, 0, @intFromEnum(operand));
    return .{ .reachable = inst.toRef() };
}

fn simpleBinOpExpr(
    zg: *ZirGen,
    ri: ResultInfo,
    node: Ast.Node.Index,
    tag: Zir.Inst.Repr.Tag,
) Error!ExprResult {
    const lhs_node, const rhs_node = zg.tree.nodeData(node).node_and_node;
    const lhs_ref = try zg.reachableExpr(.simple_rvalue, lhs_node, node);
    const rhs_ref = try zg.reachableExpr(.simple_rvalue, rhs_node, node);
    const inst = try zg.addInst(tag, .{ @intFromEnum(lhs_ref), @intFromEnum(rhs_ref) });
    return .{ .reachable = try zg.rvalue(ri, inst.toRef()) };
}

fn blockExpr(zg: *ZirGen, ri: ResultInfo, node: Ast.Node.Index) Error!ExprResult {
    const tree = zg.tree;

    var buf: [2]Ast.Node.Index = undefined;
    const lbrace = tree.nodeMainToken(node);
    const statements = zg.tree.blockStatements(&buf, node).?;

    const has_label = tree.isTokenPrecededByTags(lbrace, &.{ .identifier, .colon });
    if (!has_label) {
        return switch (try zg.blockStatements(statements)) {
            .reachable => .{ .reachable = .void_value },
            .@"unreachable" => |noreturn_node| .{ .@"unreachable" = noreturn_node },
        };
    }

    const block_inst = try zg.addInst(.block, undefined); // payload set later

    const break_ri = ri.breakInfo();

    try zg.scopes.append(zg.gpa, .{ .break_target = .{
        .label = .{
            .tok = lbrace - 2,
            .used = false,
        },
        .allow_unlabeled = false,
        .block_inst = block_inst,
        .ri = break_ri,
    } });
    defer assert(zg.scopes.pop() != null);

    switch (try zg.blockStatements(statements)) {
        .@"unreachable" => {},
        .reachable => {
            // Add the implicit `break :blk;` at the end.
            const result = try zg.rvalue(break_ri, .void_value);
            _ = try zg.addInst(.@"break", .{
                @intFromEnum(block_inst),
                @intFromEnum(result),
            });
        },
    }

    // TODO: check label used

    zg.setData(block_inst, .{
        @intCast(zg.wip_instructions.len),
        0,
    });

    // Because there is a break targeting the label, the result is not always-noreturn.
    // We propagated the result info, so no `rvalue` call.
    return .{ .reachable = block_inst.toRef() };
}

fn breakExpr(zg: *ZirGen, node: Ast.Node.Index) Error!ExprResult {
    const opt_break_label, const opt_operand_node = zg.tree.nodeData(node).opt_token_and_opt_node;
    var it = zg.iterateScopes();
    const target_block: Zir.Inst.Index, const operand_ri: ResultInfo = while (it.next()) |scope| {
        switch (scope.*) {
            .break_target => |*bt| {
                if (opt_break_label.unwrap()) |break_label| {
                    if (bt.label) |*l| {
                        if (zg.tokenIdentEql(l.tok, break_label)) {
                            l.used = true;
                            break .{ bt.block_inst, bt.ri };
                        }
                    }
                } else {
                    if (bt.allow_unlabeled) {
                        break .{ bt.block_inst, bt.ri };
                    }
                }
            },
            .namespace => {
                // TODO: label inaccessible if exists
                @panic("TODO");
            },
            .local_val => {},
        }
    } else @panic("TODO error");
    const operand: Zir.Ref = if (opt_operand_node.unwrap()) |operand_node| o: {
        break :o try zg.reachableExpr(operand_ri, operand_node, node);
    } else .void_value;
    _ = try zg.addInst(.@"break", .{
        @intFromEnum(target_block),
        @intFromEnum(operand),
    });
    return .{ .@"unreachable" = node };
}

fn structDeclInner(
    zg: *ZirGen,
    node: Ast.Node.Index,
    container_decl: Ast.full.ContainerDecl,
    layout: std.builtin.Type.ContainerLayout,
    backing_int_node: Ast.Node.OptionalIndex,
    name_strategy: Zir.NameStrategy,
) Error!Zir.Ref {
    const gpa = zg.gpa;
    const tree = zg.tree;

    for (container_decl.ast.members) |member_node| {
        const container_field = tree.fullContainerField(member_node) orelse continue;
        if (container_field.ast.tuple_like) @panic("TODO tuple field");
    }

    const struct_decl_inst = try zg.addInst(.extended, undefined); // payload set later

    // We're going to do 3 passes.
    //
    // Pass 1 (`scanContainer`):
    // * Count declarations (including unnamed)
    // * Check declaration/field names for conflicts
    // * Populate ArrayHashMap of named declarations
    // * Reserve sequential instruction indices for declarations
    //
    // Pass 2:
    // * Count fields
    // * Store field information into scratch buffer
    // * Generate field expressions
    // * Generate declarations
    // * Count how much `extra` capacity we need
    //
    // Pass 3:
    // * Populate `extra`

    var named_decls: std.AutoArrayHashMapUnmanaged(Zir.NullTerminatedString, Ast.Node.Index) = .empty;
    defer named_decls.deinit(gpa);
    const decl_count: u32 = switch (try zg.scanContainer(container_decl.ast.members, &named_decls, .@"struct")) {
        .success => |decl_count| decl_count,
        .fail => return error.AnalysisFail,
    };

    try zg.scopes.append(gpa, .{ .namespace = .{
        .members = &named_decls,
    } });
    defer assert(zg.scopes.pop().?.namespace.members == &named_decls);

    // Before the second pass, let's reserve the instructions which will become our declarations.
    var wip_decls: WipDecls = try .init(zg, decl_count);

    // Pass 2

    var fields_scratch: Scratch(struct {
        name: Zir.NullTerminatedString,
        type_body_start: Zir.Inst.Index,
        align_body_start: Zir.Inst.Index.Optional,
        init_body_start: Zir.Inst.Index.Optional,
    }) = .empty(zg);
    defer fields_scratch.free();

    var extra_len: u32 = @typeInfo(Zir.Inst.Repr.Extended.StructDecl).@"struct".fields.len;
    var fields_len: u32 = 0;

    var any_comptime_fields = false;
    var any_aligned_fields = false;
    var any_default_inits = false;

    var known_non_opv = false;
    var known_comptime_only = false;

    known_non_opv = known_non_opv; // TODO
    known_comptime_only = known_comptime_only; // TODO

    for (container_decl.ast.members) |member_node| {
        const cf = try wip_decls.containerMember(member_node) orelse continue;
        fields_len += 1;

        extra_len += 2; // `name`, `type_body_start`

        if (cf.comptime_token != null) any_comptime_fields = true;
        if (cf.ast.align_expr != .none) {
            any_aligned_fields = true;
            extra_len += 1; // `align_body_start`
        }
        if (cf.ast.value_expr != .none) {
            any_default_inits = true;
            extra_len += 1; // `init_body_start`
        }

        const field_name = try zg.identAsString(cf.ast.main_token);

        const type_body_start: Zir.Inst.Index = @enumFromInt(zg.wip_instructions.len);
        const type_ref = try zg.reachableExpr(.simple_rvalue, cf.ast.type_expr.unwrap().?, node);
        _ = try zg.addInst(.@"break", .{ @intFromEnum(struct_decl_inst), @intFromEnum(type_ref) });

        const align_body_start: Zir.Inst.Index.Optional = if (cf.ast.align_expr.unwrap()) |align_node| s: {
            const start = Zir.Inst.Index.toOptional(@enumFromInt(zg.wip_instructions.len));
            const align_ref = try zg.reachableExpr(.simple_rvalue, align_node, node);
            _ = try zg.addInst(.@"break", .{ @intFromEnum(struct_decl_inst), @intFromEnum(align_ref) });
            break :s start;
        } else .none;

        const init_body_start: Zir.Inst.Index.Optional = if (cf.ast.value_expr.unwrap()) |init_node| s: {
            const start = Zir.Inst.Index.toOptional(@enumFromInt(zg.wip_instructions.len));
            const init_ref = try zg.reachableExpr(.simple_rvalue, init_node, node);
            _ = try zg.addInst(.@"break", .{ @intFromEnum(struct_decl_inst), @intFromEnum(init_ref) });
            break :s start;
        } else .none;

        try fields_scratch.append(.{
            .name = field_name,
            .type_body_start = type_body_start,
            .align_body_start = align_body_start,
            .init_body_start = init_body_start,
        });
    }

    const backing_int_body_start: Zir.Inst.Index = if (backing_int_node.unwrap()) |n| s: {
        const start: Zir.Inst.Index = @enumFromInt(zg.wip_instructions.len);
        const backing_int_ref = try zg.reachableExpr(.simple_rvalue, n, node);
        _ = try zg.addInst(.@"break", .{ @intFromEnum(struct_decl_inst), @intFromEnum(backing_int_ref) });
        break :s start;
    } else undefined;

    const field_bit_bags = std.math.divCeil(u32, fields_len, 32) catch unreachable;
    if (any_comptime_fields) extra_len += field_bit_bags; // `field_is_comptime`
    if (any_aligned_fields) extra_len += field_bit_bags; // `field_has_align`
    if (any_default_inits) extra_len += field_bit_bags; // `field_has_default`

    const captures: []const Zir.TypeCapture = &.{}; // TODO

    const any_captures = captures.len > 0;
    const any_fields = fields_len > 0;
    const any_decls = decl_count > 0;

    if (any_captures) extra_len += 1; // `captures_len`
    if (any_fields) extra_len += 1; // `fields_len`
    if (any_decls) extra_len += 1; // `decls_len`

    extra_len += captures.len * 2; // `capture`, `capture_name`

    if (backing_int_node != .none) extra_len += 1; // `backing_int_body_start`

    const small: Zir.Inst.Repr.Extended.StructDecl.Small = .{
        .any_captures = any_captures,
        .any_fields = any_fields,
        .any_decls = any_decls,
        .any_comptime_fields = any_comptime_fields,
        .any_aligned_fields = any_aligned_fields,
        .any_default_inits = any_default_inits,
        .has_backing_int = backing_int_node != .none,
        .layout = layout,
        .name_strategy = name_strategy,
        .known_non_opv = known_non_opv,
        .known_comptime_only = known_comptime_only,
    };

    var extra = try zg.beginExtra(extra_len);

    extra.appendStruct(Zir.Inst.Repr.Extended.StructDecl, .{
        .next_inst = @enumFromInt(zg.wip_instructions.len),
        .fields_hash_0 = 0, // TODO
        .fields_hash_1 = 0, // TODO
        .fields_hash_2 = 0, // TODO
        .fields_hash_3 = 0, // TODO
        .src_line = 0, // TODO
    });

    if (any_captures) extra.append(@as(u32, @intCast(captures.len))); // `captures_len`
    if (any_fields) extra.append(fields_len); // `fields_len`
    if (any_decls) extra.append(decl_count); // `decls_len`

    extra.appendSlice(Zir.TypeCapture, captures); // `capture`
    extra.appendSlice(Zir.NullTerminatedString, &.{}); // `capture_name` TODO

    if (backing_int_node != .none) extra.append(backing_int_body_start); // `backing_int_body_start`

    if (any_comptime_fields) {
        // `field_is_comptime`
        var wip: WipFlagBits = .init(&extra);
        for (container_decl.ast.members) |member_node| {
            if (tree.fullContainerField(member_node)) |cf| {
                wip.next(cf.comptime_token != null);
            }
        }
        wip.finish();
    }
    if (any_aligned_fields) {
        // `field_has_align`
        var wip: WipFlagBits = .init(&extra);
        for (container_decl.ast.members) |member_node| {
            if (tree.fullContainerField(member_node)) |cf| {
                wip.next(cf.ast.align_expr != .none);
            }
        }
        wip.finish();
    }
    if (any_default_inits) {
        // `field_has_default`
        var wip: WipFlagBits = .init(&extra);
        for (container_decl.ast.members) |member_node| {
            if (tree.fullContainerField(member_node)) |cf| {
                wip.next(cf.ast.value_expr != .none);
            }
        }
        wip.finish();
    }

    for (0..fields_scratch.len) |field_idx| {
        const field = fields_scratch.get(field_idx);
        extra.append(field.name); // `fields.name`
        extra.append(field.type_body_start); // `fields.type_body_start`
        if (field.align_body_start.unwrap()) |x| extra.append(x); // `fields.align_body_start`
        if (field.init_body_start.unwrap()) |x| extra.append(x); // `fields.init_body_start`
    }

    zg.setExtended(struct_decl_inst, .struct_decl, @bitCast(small), @intFromEnum(extra.finish()));

    return struct_decl_inst.toRef();
}

/// This is always the first pass done over a container's members. It reports name conflicts,
/// populates `namespace_decls`, and returns the total number of declarations in the container
/// (including unnamed declarations). If this function returns successfully, the names in the
/// namespace have been successfully validated. That is:
/// * they are not `_` (`@"_"` is okay)
/// * they do not conflict with one another
/// * they do not shadow any locals
/// * they do not shadow any primitives
///
/// For test names, we have done most validation, but have not yet validated that decltests refer
/// to real declarations. That is handled by `WipDecls`, which creates the test either way.
fn scanContainer(
    zg: *ZirGen,
    members: []const Ast.Node.Index,
    decls_out: *std.AutoArrayHashMapUnmanaged(Zir.NullTerminatedString, Ast.Node.Index),
    container_kind: enum { @"struct", @"union", @"enum", @"opaque" },
) Allocator.Error!union(enum) {
    /// The namespace is valid.
    /// `decls_out` has been populated with all named declarations.
    /// This value is the total number of declarations, including unnamed.
    success: u32,
    /// There was a problem with the namespace.
    /// An error has been reported.
    /// Analysis of the container cannot continue.
    fail,
} {
    const gpa = zg.gpa;
    const tree = zg.tree;

    // Small namespaces are very common (and larger ones relatively rare), so
    // use a `StackFallbackAllocator` to avoid hitting the gpa for those.
    var sfba_state = std.heap.stackFallback(512, gpa);
    const sfba = sfba_state.get();

    // This type forms a linked list of source tokens declaring the same name.
    const NameEntry = struct {
        tok: Ast.TokenIndex,
        /// Index into `linked_entries` below. In non-error scenarios, this is
        /// always `null`, so `linked_entries` below is always empty.
        next: ?u32,
    };

    var names: std.AutoArrayHashMapUnmanaged(Zir.NullTerminatedString, NameEntry) = .empty;
    var test_names: std.AutoArrayHashMapUnmanaged(Zir.NullTerminatedString, NameEntry) = .empty;
    var decltest_names: std.AutoArrayHashMapUnmanaged(Zir.NullTerminatedString, NameEntry) = .empty;
    var linked_entries: std.ArrayListUnmanaged(NameEntry) = .empty;
    defer {
        names.deinit(sfba);
        test_names.deinit(sfba);
        decltest_names.deinit(sfba);
        linked_entries.deinit(sfba);
    }

    var any_errors = false;
    var any_duplicates = false;
    var decl_count: u32 = 0;

    for (members) |member_node| {
        const Kind = enum { decl, field };
        const kind: Kind, const name_token = switch (tree.nodeTag(member_node)) {
            .container_field_init,
            .container_field_align,
            .container_field,
            => blk: {
                var full = tree.fullContainerField(member_node).?;
                switch (container_kind) {
                    .@"struct", .@"opaque" => {},
                    .@"union", .@"enum" => full.convertToNonTupleLike(tree),
                }
                if (full.ast.tuple_like) continue; // no name
                break :blk .{ .field, full.ast.main_token };
            },

            .global_var_decl,
            .local_var_decl,
            .simple_var_decl,
            .aligned_var_decl,
            => blk: {
                decl_count += 1;
                break :blk .{ .decl, tree.nodeMainToken(member_node) + 1 };
            },

            .fn_proto_simple,
            .fn_proto_multi,
            .fn_proto_one,
            .fn_proto,
            .fn_decl,
            => blk: {
                decl_count += 1;
                const ident = tree.nodeMainToken(member_node) + 1;
                if (tree.tokenTag(ident) != .identifier) {
                    try zg.addError(.{ .node = member_node }, "missing function name", .{}, &.{});
                    any_errors = true;
                    continue;
                }
                break :blk .{ .decl, ident };
            },

            .@"comptime", .@"usingnamespace" => {
                decl_count += 1;
                continue;
            },

            .test_decl => {
                decl_count += 1;
                // We don't want shadowing detection here, and test names work a bit differently, so
                // we must do the redeclaration detection ourselves.
                const test_name_token = tree.nodeMainToken(member_node) + 1;
                const new_ent: NameEntry = .{
                    .tok = test_name_token,
                    .next = null,
                };
                switch (tree.tokenTag(test_name_token)) {
                    else => {}, // unnamed test
                    .string_literal => {
                        const name = zg.restrictedStrLitAsString(test_name_token, "test name") catch |err| switch (err) {
                            error.AnalysisFail => {
                                any_errors = true;
                                continue;
                            },
                            error.OutOfMemory => |e| return e,
                        };
                        const gop = try test_names.getOrPut(sfba, name);
                        if (gop.found_existing) {
                            const new_idx: u32 = @intCast(linked_entries.items.len);
                            try linked_entries.append(sfba, new_ent);
                            var e = gop.value_ptr;
                            while (e.next) |next_idx| e = &linked_entries.items[next_idx];
                            e.next = new_idx;
                            any_duplicates = true;
                        } else {
                            gop.value_ptr.* = new_ent;
                        }
                    },
                    .identifier => {
                        const test_name_raw = tree.tokenSlice(test_name_token);
                        if (mem.eql(u8, test_name_raw, "_")) {
                            try zg.addError(.token(test_name_token), "'_' used as an identifier without @\"_\" syntax", .{}, &.{});
                            any_errors = true;
                            continue;
                        }
                        if (std.zig.primitives.isPrimitive(test_name_raw)) {
                            try zg.addError(.token(test_name_token), "cannot test a primitive", .{}, &.{});
                            any_errors = true;
                            continue;
                        }
                        const name = zg.identAsString(test_name_token) catch |err| switch (err) {
                            error.AnalysisFail => {
                                any_errors = true;
                                continue;
                            },
                            error.OutOfMemory => |e| return e,
                        };
                        const gop = try decltest_names.getOrPut(sfba, name);
                        if (gop.found_existing) {
                            const new_idx: u32 = @intCast(linked_entries.items.len);
                            try linked_entries.append(sfba, new_ent);
                            var e = gop.value_ptr;
                            while (e.next) |next_idx| e = &linked_entries.items[next_idx];
                            e.next = new_idx;
                            any_duplicates = true;
                        } else {
                            gop.value_ptr.* = new_ent;
                        }
                    },
                }
                continue;
            },

            else => unreachable,
        };

        const name_str = zg.identAsString(name_token) catch |err| switch (err) {
            error.AnalysisFail => {
                any_errors = true;
                continue;
            },
            error.OutOfMemory => |e| return e,
        };

        {
            const gop = try names.getOrPut(sfba, name_str);
            const new_ent: NameEntry = .{
                .tok = name_token,
                .next = null,
            };
            if (gop.found_existing) {
                const new_idx: u32 = @intCast(linked_entries.items.len);
                try linked_entries.append(sfba, new_ent);
                var e = gop.value_ptr;
                while (e.next) |next_idx| e = &linked_entries.items[next_idx];
                e.next = new_idx;
                any_duplicates = true;
            } else {
                gop.value_ptr.* = new_ent;
            }
        }

        // For fields, we only needed that duplicate check; for decls we have more work to do.
        switch (kind) {
            .decl => {},
            .field => continue,
        }

        try decls_out.put(gpa, name_str, member_node);

        const name_raw = tree.tokenSlice(name_token);
        if (std.zig.primitives.isPrimitive(name_raw)) {
            try zg.addError(.token(name_token), "name shadows primitive '{s}'", .{name_raw}, &.{
                try zg.errNote(.token(name_token), "consider usng @\"{s}\" to disambiguate", .{name_raw}),
            });
            any_errors = true;
            continue;
        }

        if (mem.eql(u8, name_raw, "_")) {
            try zg.addError(.token(name_token), "'_' used as an identifier without @\"_\" syntax", .{}, &.{});
            any_errors = true;
            continue;
        }

        var scope_it = zg.iterateScopes();
        while (scope_it.next()) |scope| switch (scope.*) {
            .local_val => |local_val| {
                if (local_val.name == name_str) {
                    try zg.addError(.token(name_token), "declaration '{}' shadows {s} from outer scope", .{
                        std.zig.fmtId(zg.getString(name_str)), @tagName(local_val.id_cat),
                    }, &.{
                        try zg.errNote(.token(local_val.name_token), "previous declaration here", .{}),
                    });
                }
            },
            .namespace => {},
            .break_target => {},
        };
    }

    if (!any_duplicates) {
        if (any_errors) return .fail;
        return .{ .success = decl_count };
    }

    @panic("TODO decl name conflict crap");
}

const WipDecls = struct {
    zg: *ZirGen,
    first_decl_inst: Zir.Inst.Index,
    decl_count: u32,
    decl_idx: u32,

    /// Reserves the next `decl_count` instructions for declarations.
    fn init(zg: *ZirGen, decl_count: u32) Allocator.Error!WipDecls {
        const gpa = zg.gpa;
        const first_decl_inst: Zir.Inst.Index = @enumFromInt(zg.wip_instructions.len);
        try zg.wip_instructions.resize(gpa, zg.wip_instructions.len + decl_count);
        @memset(zg.wip_instructions.items(.tag)[@intFromEnum(first_decl_inst)..], .extended);
        return .{
            .zg = zg,
            .first_decl_inst = first_decl_inst,
            .decl_count = decl_count,
            .decl_idx = 0,
        };
    }

    /// If this is a declaration, processes it and returns `null`.
    /// Otherwise, returns the field info.
    /// Declaration names have already been validated by `scanContainer`. As such, this function
    /// never returns `error.AnalysisFail`; any issues with the declarations are converted into
    /// `zirgen_error` instructions in the decls themselves.
    fn containerMember(wip: *WipDecls, node: Ast.Node.Index) Allocator.Error!?Ast.full.ContainerField {
        const zg = wip.zg;
        const tree = zg.tree;
        switch (tree.nodeTag(node)) {
            .container_field,
            .container_field_init,
            .container_field_align,
            => return tree.fullContainerField(node).?,

            .fn_proto,
            .fn_proto_multi,
            .fn_proto_one,
            .fn_proto_simple,
            .fn_decl,
            => |node_tag| {
                const decl_inst = wip.next();

                // In case we need to fail
                const old_instructions_len: u32 = @intCast(zg.wip_instructions.len);
                const old_extra_len: u32 = @intCast(zg.wip_extra.items.len);

                var buf: [1]Ast.Node.Index = undefined;
                const full = tree.fullFnProto(&buf, node).?;
                const opt_body_node: Ast.Node.OptionalIndex = switch (node_tag) {
                    .fn_decl => tree.nodeData(node).node_and_node[1].toOptional(),
                    .fn_proto, .fn_proto_multi, .fn_proto_one, .fn_proto_simple => .none,
                    else => unreachable,
                };

                const is_pub = full.visib_token != null;
                const name_token = full.name_token.?;
                const name = zg.identAsString(name_token) catch |err| switch (err) {
                    error.AnalysisFail => unreachable, // validated by `scanContainer`
                    error.OutOfMemory => |e| return e,
                };
                assert(!mem.eql(u8, tree.tokenSlice(name_token), "_")); // validated by `scanContainer`

                const linkage: Zir.Declaration.Linkage, const inline_keyword: bool, const is_noinline: bool = l: {
                    const t = full.extern_export_inline_token orelse break :l .{ .normal, false, false };
                    break :l switch (tree.tokenTag(t)) {
                        .keyword_extern => .{ .@"extern", false, false },
                        .keyword_export => .{ .@"export", false, false },
                        .keyword_inline => .{ .normal, true, false },
                        .keyword_noinline => .{ .normal, false, true },
                        else => unreachable,
                    };
                };

                const callconv_strat: enum { expr, @"inline", auto } = cc: {
                    if (full.ast.callconv_expr.unwrap()) |callconv_node| {
                        if (inline_keyword) {
                            try zg.addError(.{ .node = callconv_node }, "explicit callconv incompatible with inline keyword", .{}, &.{});
                            return wip.failNamedDecl(decl_inst, name, is_pub, old_instructions_len, old_extra_len);
                        }
                        break :cc .expr;
                    }
                    if (inline_keyword) break :cc .@"inline";
                    break :cc .auto;
                };

                const lib_name: ?Zir.NullTerminatedString = if (full.lib_name) |lib_name_tok| lib_name: {
                    assert(linkage == .@"extern");
                    break :lib_name zg.restrictedStrLitAsString(lib_name_tok, "library name") catch |err| switch (err) {
                        error.OutOfMemory => |e| return e,
                        error.AnalysisFail => null,
                    };
                } else null;

                const return_type = full.ast.return_type.unwrap().?;
                const maybe_bang = tree.firstToken(return_type) - 1;
                const is_inferred_error = tree.tokenTag(maybe_bang) == .bang;
                switch (linkage) {
                    .normal => {},
                    .@"extern", .@"export" => if (is_inferred_error) {
                        try zg.addError(.token(maybe_bang), "{s} function may not have inferred error set", .{@tagName(linkage)}, &.{});
                        return wip.failNamedDecl(decl_inst, name, is_pub, old_instructions_len, old_extra_len);
                    },
                }
                switch (linkage) {
                    .@"extern" => {
                        assert(opt_body_node == .none); // validated by parser
                    },
                    .@"export", .normal => if (opt_body_node == .none) {
                        try zg.addError(.{ .node = node }, "non-extern function has no body", .{}, &.{});
                        return wip.failNamedDecl(decl_inst, name, is_pub, old_instructions_len, old_extra_len);
                    },
                }

                // Evaluate the non-generic bodies first. Then, we'll start adding parameters.

                const callconv_body_start: ?Zir.Inst.Index = switch (callconv_strat) {
                    .expr => try wip.exprInfallible(.simple_rvalue, full.ast.callconv_expr.unwrap().?, .{ .break_val = decl_inst }),
                    .@"inline", .auto => null,
                };

                const align_body_start: ?Zir.Inst.Index = if (full.ast.align_expr.unwrap()) |align_node|
                    try wip.exprInfallible(.simple_rvalue, align_node, .{ .break_val = decl_inst })
                else
                    null;

                const addrspace_body_start: ?Zir.Inst.Index = if (full.ast.addrspace_expr.unwrap()) |addrspace_node|
                    try wip.exprInfallible(.simple_rvalue, addrspace_node, .{ .break_val = decl_inst })
                else
                    null;

                const linksection_body_start: ?Zir.Inst.Index = if (full.ast.section_expr.unwrap()) |linksection_node|
                    try wip.exprInfallible(.simple_rvalue, linksection_node, .{ .break_val = decl_inst })
                else
                    null;

                // We will add parameters (interleaved with evaluating their types), then evaluate the return type,
                // then finally the function body.

                // The idea here is that `zg.scopes.items[old_scopes_len..]` is all our parameters.
                // That's useful because a) we need to pop them later, b) we want to iterate them
                // later to check if they were used, and c) we need to change their `ptr_strat` once
                // we get to the fn body (see `Scope.local_val.ptr_strat.make_ref_inst`).
                const old_scopes_len = zg.scopes.items.len;
                // Pop the parameter scopes once we're done!
                defer zg.scopes.shrinkRetainingCapacity(old_scopes_len);

                // This is used to determine whether a type expression referenced any parameter, in order
                // to determine whether or not it is a generic type. It is always reset to `false` after
                // a recursive call which may have modified it.
                var any_param_used: bool = false;

                // Use `scratch` to store all parameters.
                // Some of this data is already in `scopes`; however, we can't rely on all
                // parameters being in there, because some might be discarded (`_: T`).
                // It's easiest to just accept a little redundancy here.
                var params_scratch: Scratch(struct {
                    /// May be `.empty`.
                    name: Zir.NullTerminatedString,
                    placeholder_inst: Zir.Inst.Index,
                    /// `.none` means anytype
                    type_body_start: Zir.Inst.Index.Optional,
                    flags: packed struct(u32) {
                        is_comptime: bool,
                        is_noalias: bool,
                        type_is_generic: bool,
                        _: u29 = 0,
                    },
                }) = .empty(zg);
                defer params_scratch.free();

                var any_comptime_params = false;
                var any_noalias_params = false;
                var any_generic_param_ty = false;
                const is_var_args = va: {
                    var param_it = full.iterate(tree);
                    while (param_it.next()) |param| {
                        const is_anytype = if (param.anytype_ellipsis3) |tok| switch (tree.tokenTag(tok)) {
                            .keyword_anytype => true,
                            .ellipsis3 => {
                                assert(param_it.next() == null); // ellipsis must come last
                                break :va true;
                            },
                            else => unreachable,
                        } else false;

                        const is_comptime, const is_noalias = flags: {
                            const tok = param.comptime_noalias orelse break :flags .{ false, false };
                            break :flags switch (tree.tokenTag(tok)) {
                                .keyword_noalias => .{ false, true },
                                .keyword_comptime => .{ true, false },
                                else => unreachable,
                            };
                        };

                        if (is_comptime) any_comptime_params = true;
                        if (is_noalias) any_noalias_params = true;

                        const param_name: Zir.NullTerminatedString = if (param.name_token) |tok| name: {
                            if (mem.eql(u8, "_", tree.tokenSlice(tok))) break :name .empty;
                            break :name zg.identAsString(tok) catch |err| switch (err) {
                                error.OutOfMemory => |e| return e,
                                error.AnalysisFail => return wip.failNamedDecl(decl_inst, name, is_pub, old_instructions_len, old_extra_len),
                            };
                        } else name: {
                            // No parameter name. If this is an extern function, that's fine!
                            if (linkage == .@"extern") break :name .empty;
                            // Otherwise, error.
                            if (is_anytype) {
                                try zg.addError(.token(param.anytype_ellipsis3.?), "missing parameter name", .{}, &.{});
                                return wip.failNamedDecl(decl_inst, name, is_pub, old_instructions_len, old_extra_len);
                            }
                            // If the expression is an identifier, it could be intended as a name,
                            // or as a type; e.g. `fn (MyType) void` vs `fn (my_arg) void`.
                            // If it's any other syntax form, it must be meant as a type;
                            // e.g. `fn (GenericFn(u32)) void`.
                            // Emit an error based on which case we're in.
                            const type_node = param.type_expr.?;
                            const intention: enum { type, ambiguous } = i: {
                                if (tree.nodeTag(type_node) != .identifier) break :i .type;
                                const ident_str = tree.tokenSlice(tree.nodeMainToken(type_node));
                                if (std.zig.primitives.isPrimitive(ident_str)) break :i .type;
                                break :i .ambiguous;
                            };
                            switch (intention) {
                                .type => try zg.addError(.{ .node = type_node }, "missing parameter name", .{}, &.{}),
                                .ambiguous => {
                                    const ident_str = tree.tokenSlice(tree.nodeMainToken(type_node));
                                    try zg.addError(.{ .node = type_node }, "missing parameter name or type", .{}, &.{
                                        try zg.errNote(.{ .node = type_node }, "if this is a name, annotate its type '{s}: T'", .{ident_str}),
                                        try zg.errNote(.{ .node = type_node }, "if this is a type, give it a name '<name>: {s}'", .{ident_str}),
                                    });
                                },
                            }
                            return wip.failNamedDecl(decl_inst, name, is_pub, old_instructions_len, old_extra_len);
                        };

                        const type_body_start: Zir.Inst.Index.Optional, const type_is_generic = b: {
                            if (is_anytype) break :b .{ .none, false };
                            const start = try wip.exprInfallible(.simple_rvalue, param.type_expr.?, .{ .break_val = decl_inst });
                            const is_generic = any_param_used;
                            any_param_used = false;
                            break :b .{ start.toOptional(), is_generic };
                        };

                        if (type_is_generic) any_generic_param_ty = true;

                        const placeholder_inst = try zg.addExtended(.value_placeholder, 0, 0);
                        try params_scratch.append(.{
                            .name = param_name,
                            .placeholder_inst = placeholder_inst,
                            .type_body_start = type_body_start,
                            .flags = .{
                                .is_comptime = is_comptime,
                                .is_noalias = is_noalias,
                                .type_is_generic = type_is_generic,
                            },
                        });

                        if (name != .empty) {
                            const is_used_as_lvalue = zg.annotations.consts_need_ref.contains(param.name_token.?);
                            try zg.scopes.append(zg.gpa, .{
                                .local_val = .{
                                    .name = param_name,
                                    .val = placeholder_inst.toRef(),
                                    // See doc comment on `Scope.local_val.ptr_strat.make_ref_inst` for
                                    // why this is appropriate here.
                                    .ptr_strat = if (is_used_as_lvalue) .make_ref_inst else .unused,
                                    .used_or_discarded_ptr = &any_param_used,
                                    .name_token = param.name_token.?,
                                    .id_cat = .@"function parameter",
                                },
                            });
                        }
                    }
                    break :va false;
                };

                // At this point:
                // * All parameter types are evaluated
                // * Named parameters are in `scopes`
                // * `params_scratch` is fully populated
                // Our remaining tasks are to evaluate the return type and the function body.

                const ret_ty_body_start: Zir.Inst.Index = try wip.exprInfallible(.simple_rvalue, full.ast.return_type.unwrap().?, .{ .break_val = decl_inst });
                const ret_ty_is_generic = any_param_used;
                any_param_used = false;

                const fn_body_start: ?Zir.Inst.Index = b: {
                    const body_node = opt_body_node.unwrap() orelse {
                        assert(linkage == .@"extern");
                        break :b null;
                    };
                    assert(linkage != .@"extern");
                    const body_start: Zir.Inst.Index = @enumFromInt(zg.wip_instructions.len);
                    // The function body might be evaluated at runtime, so, to make sure `&param == &param`,
                    // we meed to create `ref` instructions ahead-of-time. To avoid doing this for *all*
                    // parameters (e.g. `u32` params probably aren't ever used as lvalues!), we can use
                    // `AstAnnotate`'s results to determine where it's necessary. That information is
                    // actually already in the `Scope`s of each parameter thanks to logic above; so, we just
                    // need to turn `make_ref_inst` into `ref`.
                    for (zg.scopes.items[old_scopes_len..]) |*scope| {
                        const param_scope = &scope.local_val; // we haven't added anything else
                        assert(param_scope.id_cat == .@"function parameter");
                        switch (param_scope.ptr_strat) {
                            .unused => {}, // `AstAnnotate` told us this parameter is never used as an lvalue
                            .ref => unreachable, // We never set this in the parameter loop above
                            .make_ref_inst => {
                                // `AstAnnotate` told us the parameter *is* used as an lvalue.
                                // Now that we're within the body, we need a single, canonical `ref`
                                // to ensure `&param == &param`.
                                const ref_inst = try zg.addExtended(.ref, 0, @intFromEnum(param_scope.val));
                                param_scope.ptr_strat = .{ .ref = ref_inst.toRef() };
                            },
                        }
                    }
                    const after_refs: Zir.Inst.Index = @enumFromInt(zg.wip_instructions.len);
                    assert(try wip.exprInfallible(.simple_rvalue, body_node, .ret_implicit) == after_refs);
                    break :b body_start;
                };

                // We've collected all our data! Final error check for unused parameters...
                try zg.checkUsed(old_scopes_len);

                // Finally, let's actually build the instruction.
                const params_len: u32 = @intCast(params_scratch.len);
                const param_bit_bags = std.math.divCeil(u32, params_len, 32) catch unreachable;
                var extra = try zg.beginExtra(@intCast(
                    @typeInfo(Zir.Inst.Repr.Extended.DeclFn).@"struct".fields.len +
                        @intFromBool(align_body_start != null) + // `align_body_start`
                        @intFromBool(linksection_body_start != null) + // `linksection_body_start`
                        @intFromBool(addrspace_body_start != null) + // `addrspace_body_start`
                        @intFromBool(callconv_body_start != null) + // `callconv_body_start`
                        @intFromBool(lib_name != null) + // `lib_name`
                        @intFromBool(linkage != .@"extern") + // `body_start`
                        (params_len * 3) + // `param`
                        (if (any_comptime_params) param_bit_bags else 0) + // `param_is_comptime`
                        (if (any_noalias_params) param_bit_bags else 0) + // `param_is_noalias`
                        (if (any_generic_param_ty) param_bit_bags else 0), // `param_ty_is_generic`
                ));
                extra.appendStruct(Zir.Inst.Repr.Extended.DeclFn, .{
                    .src_hash_0 = 0, // TODO
                    .src_hash_1 = 0, // TODO
                    .src_hash_2 = 0, // TODO
                    .src_hash_3 = 0, // TODO
                    .proto_hash_0 = 0, // TODO
                    .proto_hash_1 = 0, // TODO
                    .proto_hash_2 = 0, // TODO
                    .proto_hash_3 = 0, // TODO
                    .src_line = 0, // TODO
                    .src_column = 0, // TODO
                    .name = name,
                    .ret_ty_body_start = ret_ty_body_start,
                    .params_len = params_len,
                });
                if (align_body_start) |s| extra.append(s);
                if (linksection_body_start) |s| extra.append(s);
                if (addrspace_body_start) |s| extra.append(s);
                if (callconv_body_start) |s| extra.append(s);
                if (lib_name) |n| extra.append(n);
                if (fn_body_start) |s| {
                    assert(linkage != .@"extern");
                    extra.append(s);
                } else {
                    assert(linkage == .@"extern");
                }
                for (0..params_len) |param_idx| {
                    const param = params_scratch.get(param_idx);
                    extra.append(param.name);
                    extra.append(param.type_body_start);
                    extra.append(param.placeholder_inst);
                }
                if (any_comptime_params) {
                    var flags: WipFlagBits = .init(&extra);
                    for (0..params_len) |i| flags.next(params_scratch.get(i).flags.is_comptime);
                    flags.finish();
                }
                if (any_noalias_params) {
                    var flags: WipFlagBits = .init(&extra);
                    for (0..params_len) |i| flags.next(params_scratch.get(i).flags.is_noalias);
                    flags.finish();
                }
                if (any_generic_param_ty) {
                    var flags: WipFlagBits = .init(&extra);
                    for (0..params_len) |i| flags.next(params_scratch.get(i).flags.type_is_generic);
                    flags.finish();
                }
                zg.setExtended(decl_inst, .decl_fn, @bitCast(@as(Zir.Inst.Repr.Extended.DeclFn.Small, .{
                    .is_pub = is_pub,
                    .has_align = align_body_start != null,
                    .has_linksection = linksection_body_start != null,
                    .has_addrspace = addrspace_body_start != null,
                    .has_callconv = callconv_body_start != null,
                    .callconv_inline = switch (callconv_strat) {
                        .auto, .expr => false,
                        .@"inline" => true,
                    },
                    .any_comptime_params = any_comptime_params,
                    .any_noalias_params = any_noalias_params,
                    .any_generic_param_ty = any_generic_param_ty,
                    .ret_ty_is_generic = ret_ty_is_generic,
                    .is_inferred_error = is_inferred_error,
                    .is_var_args = is_var_args,
                    .is_noinline = is_noinline,
                    .linkage = linkage,
                    .has_lib_name = lib_name != null,
                })), @intFromEnum(extra.finish()));
                return null;
            },

            .global_var_decl,
            .local_var_decl,
            .simple_var_decl,
            .aligned_var_decl,
            => {
                const decl_inst = wip.next();

                // In case we need to fail
                const old_instructions_len: u32 = @intCast(zg.wip_instructions.len);
                const old_extra_len: u32 = @intCast(zg.wip_extra.items.len);

                const full = tree.fullVarDecl(node).?;

                const is_const = switch (tree.tokenTag(full.ast.mut_token)) {
                    .keyword_const => true,
                    .keyword_var => false,
                    else => unreachable,
                };
                const is_pub = full.visib_token != null;

                const name_token = full.ast.mut_token + 1;
                const name = zg.identAsString(name_token) catch |err| switch (err) {
                    error.AnalysisFail => unreachable, // validated by `scanContainer`
                    error.OutOfMemory => |e| return e,
                };
                assert(!mem.eql(u8, tree.tokenSlice(name_token), "_")); // validated by `scanContainer`

                const is_threadlocal = if (full.threadlocal_token) |t| tl: {
                    if (is_const) {
                        try zg.addError(.token(t), "threadlocal variable cannot be constant", .{}, &.{});
                        return wip.failNamedDecl(decl_inst, name, is_pub, old_instructions_len, old_extra_len);
                    }
                    break :tl true;
                } else false;

                const linkage: Zir.Declaration.Linkage = if (full.extern_export_token) |t| switch (tree.tokenTag(t)) {
                    .keyword_export => .@"export",
                    .keyword_extern => .@"extern",
                    else => unreachable,
                } else .normal;

                const lib_name: ?Zir.NullTerminatedString = if (full.lib_name) |lib_name_tok| lib_name: {
                    assert(linkage == .@"extern");
                    break :lib_name zg.restrictedStrLitAsString(lib_name_tok, "library name") catch |err| switch (err) {
                        error.OutOfMemory => |e| return e,
                        error.AnalysisFail => null,
                    };
                } else null;

                // Always `null` for `linkage == .@"extern"`. Always non-`null` for other `linkage`.
                const init_body_start: ?Zir.Inst.Index = switch (linkage) {
                    .@"extern" => b: {
                        if (full.ast.init_node.unwrap()) |init_node| {
                            try zg.addError(.{ .node = init_node }, "extern variables have no initializers", .{}, &.{});
                            return wip.failNamedDecl(decl_inst, name, is_pub, old_instructions_len, old_extra_len);
                        }
                        break :b null;
                    },
                    .normal, .@"export" => b: {
                        if (full.ast.init_node.unwrap()) |init_node| {
                            break :b try wip.exprInfallible(.simple_rvalue, init_node, .{ .break_val = decl_inst });
                        } else {
                            try zg.addError(.{ .node = node }, "variables must be initialized", .{}, &.{});
                            return wip.failNamedDecl(decl_inst, name, is_pub, old_instructions_len, old_extra_len);
                        }
                    },
                };

                // Always set for `linkage == .@"extern"`.
                const type_body_start: ?Zir.Inst.Index = b: {
                    if (full.ast.type_node.unwrap()) |type_node| {
                        break :b try wip.exprInfallible(.simple_rvalue, type_node, .{ .break_val = decl_inst });
                    }
                    if (linkage == .@"extern") {
                        try zg.addError(.{ .node = node }, "unable to infer variable type", .{}, &.{});
                        return wip.failNamedDecl(decl_inst, name, is_pub, old_instructions_len, old_extra_len);
                    }
                    break :b null;
                };

                const align_body_start: ?Zir.Inst.Index = if (full.ast.align_node.unwrap()) |align_node|
                    try wip.exprInfallible(.simple_rvalue, align_node, .{ .break_val = decl_inst })
                else
                    null;

                const addrspace_body_start: ?Zir.Inst.Index = if (full.ast.addrspace_node.unwrap()) |addrspace_node|
                    try wip.exprInfallible(.simple_rvalue, addrspace_node, .{ .break_val = decl_inst })
                else
                    null;

                const linksection_body_start: ?Zir.Inst.Index = if (full.ast.section_node.unwrap()) |linksection_node|
                    try wip.exprInfallible(.simple_rvalue, linksection_node, .{ .break_val = decl_inst })
                else
                    null;

                var extra = try zg.beginExtra(@intCast(
                    @typeInfo(Zir.Inst.Repr.Extended.DeclConstVar).@"struct".fields.len +
                        @intFromBool(init_body_start != null) +
                        @intFromBool(type_body_start != null) +
                        @intFromBool(align_body_start != null) +
                        @intFromBool(linksection_body_start != null) +
                        @intFromBool(addrspace_body_start != null) +
                        @intFromBool(lib_name != null),
                ));
                extra.appendStruct(Zir.Inst.Repr.Extended.DeclConstVar, .{
                    .src_hash_0 = 0, // TODO
                    .src_hash_1 = 0, // TODO
                    .src_hash_2 = 0, // TODO
                    .src_hash_3 = 0, // TODO
                    .src_line = 0, // TODO
                    .src_column = 0, // TODO
                    .name = name,
                });
                if (init_body_start) |s| extra.append(s);
                if (type_body_start) |s| extra.append(s);
                if (align_body_start) |s| extra.append(s);
                if (linksection_body_start) |s| extra.append(s);
                if (addrspace_body_start) |s| extra.append(s);
                if (lib_name) |n| extra.append(n);
                zg.setExtended(decl_inst, if (is_const) .decl_const else .decl_var, @bitCast(@as(Zir.Inst.Repr.Extended.DeclConstVar.Small, .{
                    .is_pub = is_pub,
                    .is_threadlocal = is_threadlocal,
                    .has_type = type_body_start != null,
                    .has_align = align_body_start != null,
                    .has_linksection = linksection_body_start != null,
                    .has_addrspace = addrspace_body_start != null,
                    .linkage = linkage,
                    .has_lib_name = lib_name != null,
                })), @intFromEnum(extra.finish()));
                return null;
            },

            .@"comptime" => {
                const decl_inst = wip.next();

                const body_start = try wip.exprInfallible(.simple_rvalue, tree.nodeData(node).node, .{ .break_void = decl_inst });

                var extra = try zg.beginExtra(@typeInfo(Zir.Inst.Repr.Extended.DeclSimple).@"struct".fields.len);
                extra.appendStruct(Zir.Inst.Repr.Extended.DeclSimple, .{
                    .src_hash_0 = 0, // TODO
                    .src_hash_1 = 0, // TODO
                    .src_hash_2 = 0, // TODO
                    .src_hash_3 = 0, // TODO
                    .src_line = 0, // TODO
                    .src_column = 0, // TODO
                    .body_start = body_start,
                });
                zg.setExtended(decl_inst, .decl_comptime, 0, @intFromEnum(extra.finish()));
                return null;
            },

            .@"usingnamespace" => {
                const decl_inst = wip.next();

                const is_pub = tree.isTokenPrecededByTags(tree.nodeMainToken(node), &.{.keyword_pub});
                const body_start = try wip.exprInfallible(.simple_rvalue, tree.nodeData(node).node, .{ .break_val = decl_inst });

                var extra = try zg.beginExtra(@typeInfo(Zir.Inst.Repr.Extended.DeclSimple).@"struct".fields.len);
                extra.appendStruct(Zir.Inst.Repr.Extended.DeclSimple, .{
                    .src_hash_0 = 0, // TODO
                    .src_hash_1 = 0, // TODO
                    .src_hash_2 = 0, // TODO
                    .src_hash_3 = 0, // TODO
                    .src_line = 0, // TODO
                    .src_column = 0, // TODO
                    .body_start = body_start,
                });
                zg.setExtended(decl_inst, .decl_usingnamespace, @intFromBool(is_pub), @intFromEnum(extra.finish()));
                return null;
            },

            .test_decl => {
                const decl_inst = wip.next();

                const opt_name_token, const body_node = tree.nodeData(node).opt_token_and_node;

                const test_name: Zir.NullTerminatedString, const is_decltest: bool =
                    if (opt_name_token.unwrap()) |name_token| switch (tree.tokenTag(name_token)) {
                        .string_literal => .{
                            zg.restrictedStrLitAsString(name_token, "test name") catch |err| switch (err) {
                                error.AnalysisFail => unreachable, // validated by `scanContainer`
                                error.OutOfMemory => |e| return e,
                            },
                            false,
                        },
                        .identifier => name: {
                            const name = zg.identAsString(name_token) catch |err| switch (err) {
                                error.AnalysisFail => unreachable, // validated by `scanContainer`
                                error.OutOfMemory => |e| return e,
                            };

                            // We need to validate that `name` actually refers to a declaration.
                            // `WipDecls` didn't check that for us, because it didn't know the full namespace contents yet!
                            // We'll create the test either way though; it doesn't do any harm to analyze a test with a bad name.
                            var it = zg.iterateScopes();
                            var found_decl: ?Ast.Node.Index = null;
                            while (it.next()) |s| switch (s.*) {
                                .local_val => |*lv| {
                                    if (lv.name == name) {
                                        try zg.addError(.token(name_token), "cannot test a {s}", .{@tagName(lv.id_cat)}, &.{
                                            try zg.errNote(.token(lv.name_token), "{s} declared here", .{@tagName(lv.id_cat)}),
                                        });
                                        if (lv.used == .none) lv.used = .fromToken(name_token);
                                        if (lv.used_or_discarded_ptr) |p| p.* = true;
                                        break :name .{ name, true };
                                    }
                                },
                                .namespace => |*ns| {
                                    if (ns.members.get(name)) |n| {
                                        if (found_decl) |f| {
                                            try zg.addError(.token(name_token), "ambiguous reference", .{}, &.{
                                                try zg.errNote(.{ .node = f }, "declared here", .{}),
                                                try zg.errNote(.{ .node = n }, "also declared here", .{}),
                                            });
                                            break :name .{ name, true };
                                        } else {
                                            found_decl = n;
                                        }
                                    }
                                },
                                .break_target => {},
                            };
                            if (found_decl == null) {
                                try zg.addError(.token(name_token), "use of undeclared identifier '{}'", .{std.zig.fmtId(zg.getString(name))}, &.{});
                            }
                            break :name .{ name, true };
                        },
                        else => unreachable,
                    } else .{ .empty, false };

                const body_start = try wip.exprInfallible(.simple_rvalue, body_node, .ret_implicit);

                if (test_name == .empty) {
                    assert(opt_name_token == .none);
                    var extra = try zg.beginExtra(@typeInfo(Zir.Inst.Repr.Extended.DeclSimple).@"struct".fields.len);
                    extra.appendStruct(Zir.Inst.Repr.Extended.DeclSimple, .{
                        .src_hash_0 = 0, // TODO
                        .src_hash_1 = 0, // TODO
                        .src_hash_2 = 0, // TODO
                        .src_hash_3 = 0, // TODO
                        .src_line = 0, // TODO
                        .src_column = 0, // TODO
                        .body_start = body_start,
                    });
                    zg.setExtended(decl_inst, .decl_unnamed_test, 0, @intFromEnum(extra.finish()));
                    return null;
                } else {
                    assert(opt_name_token != .none);
                    var extra = try zg.beginExtra(@typeInfo(Zir.Inst.Repr.Extended.DeclNamedTest).@"struct".fields.len);
                    extra.appendStruct(Zir.Inst.Repr.Extended.DeclNamedTest, .{
                        .src_hash_0 = 0, // TODO
                        .src_hash_1 = 0, // TODO
                        .src_hash_2 = 0, // TODO
                        .src_hash_3 = 0, // TODO
                        .src_line = 0, // TODO
                        .src_column = 0, // TODO
                        .name = test_name,
                        .body_start = body_start,
                    });
                    zg.setExtended(decl_inst, if (is_decltest) .decl_decltest else .decl_named_test, 0, @intFromEnum(extra.finish()));
                    return null;
                }
            },

            else => unreachable,
        }
    }

    /// Internal function; gets the `Inst.Index` of the next decl, and increments the internal index.
    fn next(wip: *WipDecls) Zir.Inst.Index {
        assert(wip.decl_idx < wip.decl_count);
        const inst: Zir.Inst.Index = @enumFromInt(@intFromEnum(wip.first_decl_inst) + wip.decl_idx);
        wip.decl_idx += 1;
        return inst;
    }
    /// Internal function; analyzes a node with `expr`, but if a fatal error occurs, replaces the
    /// body with `zirgen_error`. Silently allows always-noreturn expressions, since they're allowed
    /// in the root of declarations.
    fn exprInfallible(
        wip: *WipDecls,
        ri: ResultInfo,
        node: Ast.Node.Index,
        exit_strat: union(enum) {
            ret_implicit,
            break_val: Zir.Inst.Index,
            break_void: Zir.Inst.Index,
        },
    ) Allocator.Error!Zir.Inst.Index {
        const zg = wip.zg;

        // If there's an error evaluating this body, we'll replace it.
        // We'll want to delete the instructions we generated in it, since they're useless to us.
        // However, we must not revert `string_bytes`, because the new error will reference it!
        const old_instructions_len = zg.wip_instructions.len;
        const old_extra_len = zg.wip_extra.items.len;

        if (zg.expr(ri, node)) |res| switch (res) {
            .@"unreachable" => {},
            .reachable => |ref| switch (exit_strat) {
                .break_val => |target| _ = try zg.addInst(.@"break", .{ @intFromEnum(target), @intFromEnum(ref) }),
                .break_void => |target| {
                    try zg.ensureResultUsed(ref);
                    _ = try zg.addInst(.@"break", .{ @intFromEnum(target), @intFromEnum(Zir.Ref.void_value) });
                },
                .ret_implicit => {
                    try zg.ensureResultUsed(ref);
                    _ = try zg.addExtended(.ret_implicit, 0, 0);
                },
            },
        } else |err| switch (err) {
            error.OutOfMemory => |e| return e,
            error.AnalysisFail => {
                zg.wip_instructions.shrinkRetainingCapacity(old_instructions_len);
                zg.wip_extra.shrinkRetainingCapacity(old_extra_len);
                _ = try zg.addExtended(.zirgen_error, 0, 0);
            },
        }

        return @enumFromInt(old_instructions_len); // == index of first instruction we added
    }

    /// Internal function; when there's something very wrong with a named decl, this function is used
    /// to write a simple failed `const` decl over it. This is OK for named decls because incremental
    /// updates correlate them solely based on name, so we won't break the instruction mapping.
    /// Has the same return type as `containerMember` for convenience; always returns `null`.
    fn failNamedDecl(
        wip: *WipDecls,
        decl_inst: Zir.Inst.Index,
        name: Zir.NullTerminatedString,
        is_pub: bool,
        old_instructions_len: u32,
        old_extra_len: u32,
    ) Allocator.Error!?Ast.full.ContainerField {
        const zg = wip.zg;

        zg.wip_instructions.shrinkRetainingCapacity(old_instructions_len);
        zg.wip_extra.shrinkRetainingCapacity(old_extra_len);

        const body_start = try zg.addExtended(.zirgen_error, 0, 0);

        var extra = try zg.beginExtra(@intCast(
            @typeInfo(Zir.Inst.Repr.Extended.DeclConstVar).@"struct".fields.len +
                1, // init_body
        ));
        extra.appendStruct(Zir.Inst.Repr.Extended.DeclConstVar, .{
            // The source hash doesn't matter; it's fine for it to clash with different failures,
            // since `Sema` doesn't give a shit how the `ZirGen` error happened. All that matters
            // is that it's distinct from successes.
            .src_hash_0 = 0,
            .src_hash_1 = 0,
            .src_hash_2 = 0,
            .src_hash_3 = 0,
            // Likewise, the source location doesn't matter.
            .src_line = 0,
            .src_column = 0,
            // We do need the name to be correct, though! Other code relies on this namespace having
            // this name, plus instruction mapping would lose the declaration if we didn't do this.
            .name = name,
        });
        extra.append(body_start);
        zg.setExtended(decl_inst, .decl_const, @bitCast(@as(Zir.Inst.Repr.Extended.DeclConstVar.Small, .{
            .is_pub = is_pub, // this is property of the namespace, so should be kept intact
            .is_threadlocal = false, // doesn't matter
            .has_type = false,
            .has_align = false,
            .has_linksection = false,
            .has_addrspace = false,
            .linkage = .normal, // doesn't matter, but if it were .@"extern" we'd need different bodies
            .has_lib_name = false,
        })), @intFromEnum(extra.finish()));
        return null;
    }
};

fn identAsString(zg: *ZirGen, ident_tok: Ast.TokenIndex) Error!Zir.NullTerminatedString {
    const gpa = zg.gpa;
    const tree = zg.tree;
    const string_bytes = &zg.wip_string_bytes;

    assert(tree.tokenTag(ident_tok) == .identifier);
    const ident_raw = tree.tokenSlice(ident_tok);

    // The identifier could be using `@"..."` syntax, so we need to parse it into a buffer
    // anyway (i.e. we can't necessarily use `ident_raw`). Given that, let's just put those
    // bytes into `string_bytes` preemtively, use them for the gop, and pop them off if the
    // entry turns out to already exist.
    const start: u32 = @intCast(string_bytes.items.len);
    if (!mem.startsWith(u8, ident_raw, "@")) {
        try string_bytes.ensureUnusedCapacity(gpa, ident_raw.len + 1); // null terminator
        string_bytes.appendSliceAssumeCapacity(ident_raw);
        string_bytes.appendAssumeCapacity(0);
    } else {
        try zg.parseStrLit(ident_tok, string_bytes, 1);
        const key = string_bytes.items[start..];
        if (mem.indexOfScalar(u8, key, 0) != null) {
            return zg.fail(.token(ident_tok), "identifier cannot contain null bytes", .{}, &.{});
        } else if (key.len == 0) {
            return zg.fail(.token(ident_tok), "identifier cannot be empty", .{}, &.{});
        }
        try string_bytes.append(gpa, 0); // null terminator
    }

    assert(string_bytes.items[string_bytes.items.len - 1] == 0); // null terminator

    const key: []const u8 = string_bytes.items[start .. string_bytes.items.len - 1]; // do not include terminator
    const gop = try zg.string_table.getOrPutContextAdapted(gpa, key, std.hash_map.StringIndexAdapter{
        .bytes = string_bytes,
    }, std.hash_map.StringIndexContext{
        .bytes = string_bytes,
    });
    errdefer comptime unreachable;

    if (gop.found_existing) {
        string_bytes.shrinkRetainingCapacity(start);
        return @enumFromInt(gop.key_ptr.*);
    } else {
        gop.key_ptr.* = start;
        return @enumFromInt(start);
    }
}
fn strLitAsString(zg: *ZirGen, str_tok: Ast.TokenIndex) Error!struct {
    index: u32,
    len: u32,
} {
    const string_bytes = &zg.wip_string_bytes;
    const gpa = zg.gpa;

    const start: u32 = @intCast(string_bytes.items.len);
    try zg.parseStrLit(str_tok, string_bytes, 0);
    const key: []const u8 = string_bytes.items[start..];
    const len: u32 = @intCast(key.len);

    if (mem.indexOfScalar(u8, key, 0) != null) {
        // The string contains a 0 byte, so index-len is necessary.
        return .{ .index = start, .len = len };
    }

    // We can use a NullTerminatedString.
    try string_bytes.append(gpa, 0);

    const gop = try zg.string_table.getOrPutContextAdapted(gpa, key, std.hash_map.StringIndexAdapter{
        .bytes = string_bytes,
    }, std.hash_map.StringIndexContext{
        .bytes = string_bytes,
    });
    errdefer comptime unreachable;

    if (gop.found_existing) {
        string_bytes.shrinkRetainingCapacity(start);
        return .{ .index = gop.key_ptr.*, .len = len };
    } else {
        gop.key_ptr.* = start;
        return .{ .index = start, .len = len };
    }
}

fn restrictedStrLitAsString(
    zg: *ZirGen,
    str_tok: Ast.TokenIndex,
    /// error messages will read e.g. "<thing> cannot contain null bytes"
    thing: []const u8,
) Error!Zir.NullTerminatedString {
    const gpa = zg.gpa;
    const string_bytes = &zg.wip_string_bytes;

    const start: u32 = @intCast(string_bytes.items.len);

    try zg.parseStrLit(str_tok, string_bytes, 0);
    const key: []const u8 = string_bytes.items[start..];
    if (mem.indexOfScalar(u8, key, 0) != null) {
        return zg.fail(.token(str_tok), "{s} cannot contain null bytes", .{thing}, &.{});
    } else if (key.len == 0) {
        return zg.fail(.token(str_tok), "{s} cannot be empty", .{thing}, &.{});
    }
    try string_bytes.append(gpa, 0); // null terminator; note that `key` does not include this!

    const gop = try zg.string_table.getOrPutContextAdapted(gpa, key, std.hash_map.StringIndexAdapter{
        .bytes = string_bytes,
    }, std.hash_map.StringIndexContext{
        .bytes = string_bytes,
    });
    errdefer comptime unreachable;

    if (gop.found_existing) {
        string_bytes.shrinkRetainingCapacity(start);
        return @enumFromInt(gop.key_ptr.*);
    } else {
        gop.key_ptr.* = start;
        return @enumFromInt(start);
    }
}

fn parseStrLit(
    zg: *ZirGen,
    tok: Ast.TokenIndex,
    /// Allocated into `zg.gpa`. The result is appended to this buffer.
    buf: *std.ArrayListUnmanaged(u8),
    /// If this is 1, the first byte of `tok` is skipped.
    /// This is used to parse `@"..."` syntax.
    offset: u1,
) Error!void {
    const gpa = zg.gpa;
    const tree = zg.tree;
    const raw_tok = tree.tokenSlice(tok);
    const raw_string = raw_tok[offset..];
    switch (try std.zig.string_literal.parseWrite(buf.writer(gpa), raw_string)) {
        .success => return,
        .failure => @panic("TODO error"),
    }
}

fn tokenIdentEql(
    zg: *ZirGen,
    tok1: Ast.TokenIndex,
    tok2: Ast.TokenIndex,
) bool {
    // TODO: do this properly!
    const tree = zg.tree;
    return std.mem.eql(u8, tree.tokenSlice(tok1), tree.tokenSlice(tok2));
}

const ResultInfo = struct {
    const simple_rvalue: ResultInfo = .{
        .is_direct_discard = false,
        .eval_mode = .rvalue,
        .ty = .none,
        .loc = .none,
    };

    /// If `true`, this expression is the RHS of `_ = x`, without any intermediate expressions.
    /// This should be treated as discarding a variable (to suppress "unused variable" errors).
    /// This is not the same thing as `loc == .discard`, since that applies recursively to
    /// expressions like `_ = .{x};`.
    is_direct_discard: bool,
    /// Whether to evaluate the expression as an rvalue, or an lvalue, in which case a pointer is returned.
    /// `ty` and `loc` will refer to that pointer value, not the corresponding rvalue.
    eval_mode: enum {
        /// Evaluate the expression as an rvalue.
        rvalue,
        /// Evaluate the expression as an lvalue. The pointer might be stored to; this should suppress "local variable is never mutated".
        lvalue_mutable,
        /// Evaluate the expression as an lvalue. The pointer will never be stored to; this does not suppress "local variable is never mutated".
        lvalue_immutable,
    },
    /// Information about the result type.
    /// Specifically, this defines the type that the result will have before it is stored to `loc`.
    ty: union(enum) {
        /// There is no result type.
        none,
        /// There is a result type. The result must be coerced to this type before being stored in `loc`.
        coerce: Zir.Ref,
        /// There is a result type, but the coercion is already handled by other logic.
        /// The type is provided only in case it is required for e.g. a casting builtin.
        implicit: Zir.Ref,
        /// There is a result type, which is known from the result location; `loc == .ptr`.
        /// The coercion will be handled by the store to the result location.
        /// If the result type is required for e.g. a casting builtin, it should be determined from the result pointer.
        implicit_from_loc,
    },
    /// After coercing the result, this union determines what to do with it.
    loc: union(enum) {
        /// There is no specific thing to do with the result; return it through `ZirGen.expr`.
        none,
        /// The result is unused; if evalating it does not have side effects (including compile
        /// errors), it is allowed for it to generate no ZIR at all.
        /// `ZirGen.expr` will return `.none`.
        discard,
        /// The result should be stored to this (typed) pointer.
        /// `ZirGen.expr` will return `.none`.
        ptr: Zir.Ref,
        /// The result should be destructured, and the components handled with `components`.
        /// `ZirGen.expr` will return `.none`.
        destructure: struct {
            src_node: Ast.Node.Index,
            components: []const DestructureComponent,
        },
    },

    const DestructureComponent = union(enum) {
        /// This component of the destructure should be stored to this typed pointer.
        typed_ptr: Zir.Ref,
        /// This component of the destructure should be stored to this inferred pointer.
        inferred_ptr: Zir.Ref,
        /// This component of the destructure should be ignored entirely, as it is discarded (assigned to `_`).
        discard,
    };

    /// Given the `ResultInfo` of a block, returns the `ResultInfo` which should be applied to a
    /// `break` targeting that block.
    fn breakInfo(ri: ResultInfo) ResultInfo {
        return .{
            // `_ = blk: { break :blk foo };` is not considered a direct discard of `foo`.
            .is_direct_discard = false,
            .eval_mode = ri.eval_mode,
            .ty = switch (ri.loc) {
                // In these cases, we are discarding the result or storing it to a result
                // pointer, so it's okay for the coercion to remain as it was.
                .discard, .ptr, .destructure => ri.ty,
                // However, if there is no result location, we need to coerce before returning
                // the operand from the block, avoiding invoking PTR.
                .none => switch (ri.ty) {
                    .none => .none,
                    .coerce, .implicit => |ty| .{ .coerce = ty },
                    .implicit_from_loc => unreachable, // `loc == .none`
                },
            },
            .loc = ri.loc,
        };
    }

    /// Returns the result type of this expression, or `null` is `ri` does not have a known result type.
    /// The result type is "pre-ref"; i.e. the result type of `e` in `@as(*u32, &e)` is `u32`, not `*u32`,
    /// despite `eval_mode == .lvalue_mutable`.
    fn resultType(ri: ResultInfo, zg: *ZirGen) Allocator.Error!?Zir.Ref {
        return switch (ri.ty) {
            .none => null,
            .coerce, .implicit => |ty| switch (ri.eval_mode) {
                .rvalue => ty,
                .lvalue_mutable, .lvalue_immutable => (try zg.addExtended(.elem_type, 0, @intFromEnum(ty))).toRef(),
            },
            .implicit_from_loc => {
                const ptr_ty = (try zg.addExtended(.typeof, 0, @intFromEnum(ri.loc.ptr))).toRef();
                // `ptr_ty` is the type of the result location; its child type is the result type.
                const ty = (try zg.addExtended(.elem_type, 0, @intFromEnum(ptr_ty))).toRef();
                // We want the result type pre-ref, so depending on `eval_mode`, we might need *another* `elem_type` instruction.
                return switch (ri.eval_mode) {
                    .rvalue => ty,
                    .lvalue_mutable, .lvalue_immutable => (try zg.addExtended(.elem_type, 0, @intFromEnum(ty))).toRef(),
                };
            },
        };
    }
};

fn rvalue(zg: *ZirGen, ri: ResultInfo, byval: Zir.Ref) Allocator.Error!Zir.Ref {
    // Early return: if discarding, we might be able to skip this work
    if (ri.loc == .discard and ri.ty == .none) return .none;

    const maybe_byref: Zir.Ref = switch (ri.eval_mode) {
        .rvalue => byval,
        .lvalue_mutable, .lvalue_immutable => (try zg.addExtended(.ref, 0, @intFromEnum(byval))).toRef(),
    };
    return zg.applyResultTypeLocation(ri, maybe_byref);
}
fn applyResultTypeLocation(zg: *ZirGen, ri: ResultInfo, uncoerced: Zir.Ref) Allocator.Error!Zir.Ref {
    const coerced: Zir.Ref = switch (ri.ty) {
        .none, .implicit, .implicit_from_loc => uncoerced,
        .coerce => |dest_ty| try zg.coerce(uncoerced, dest_ty),
    };
    switch (ri.loc) {
        .none => return coerced,
        .discard => return .none,
        .ptr => |dest_ptr| {
            _ = try zg.addInst(.store, .{
                @intFromEnum(dest_ptr),
                @intFromEnum(coerced),
            });
            return .none;
        },
        .destructure => |destructure| {
            _ = destructure;
            @panic("TODO");
        },
    }
}

fn ensureResultUsed(zg: *ZirGen, result: Zir.Ref) Allocator.Error!void {
    if (result == .void_value) return;
    // TODO: more sophisticated checks
    _ = try zg.addExtended(.ensure_result_used, 0, @intFromEnum(result));
}

const ErrorNoteIndex = enum(u32) { _ };
fn errNote(zg: *ZirGen, src: Zir.CompileError.Src, comptime format: []const u8, args: anytype) Allocator.Error!ErrorNoteIndex {
    const gpa = zg.gpa;

    const msg_idx: u32 = @intCast(zg.wip_string_bytes.items.len);
    const writer = zg.wip_string_bytes.writer(gpa);
    try writer.print(format, args);
    try writer.writeByte(0);

    try zg.wip_error_notes.append(gpa, .{
        .msg = @enumFromInt(msg_idx),
        .src = .init(src),
    });

    return @enumFromInt(zg.wip_error_notes.items.len - 1);
}
fn addError(zg: *ZirGen, src: Zir.CompileError.Src, comptime format: []const u8, args: anytype, notes: []const ErrorNoteIndex) Allocator.Error!void {
    const gpa = zg.gpa;

    const first_note: u32 = if (notes.len > 0) first: {
        for (notes[0 .. notes.len - 1], notes[1..]) |prev, next| {
            assert(@intFromEnum(next) == @intFromEnum(prev) - 1);
        }
        assert(@intFromEnum(notes[0]) + notes.len == zg.wip_error_notes.items.len);
        break :first @intFromEnum(notes[0]);
    } else 0;

    const msg_idx: u32 = @intCast(zg.wip_string_bytes.items.len);
    const writer = zg.wip_string_bytes.writer(gpa);
    try writer.print(format, args);
    try writer.writeByte(0);

    try zg.wip_compile_errors.append(gpa, .{
        .msg = @enumFromInt(msg_idx),
        .src = .init(src),
        .first_note = first_note,
        .note_count = @intCast(notes.len),
    });
}
fn fail(zg: *ZirGen, src: Zir.CompileError.Src, comptime format: []const u8, args: anytype, notes: []const ErrorNoteIndex) Error {
    try zg.addError(src, format, args, notes);
    return error.AnalysisFail;
}
fn failWithNumberError(zg: *ZirGen, err: std.zig.number_literal.Error, tok: Ast.TokenIndex, bytes: []const u8) Error {
    const is_float = mem.indexOfScalar(u8, bytes, '.') != null;
    switch (err) {
        .leading_zero => if (is_float) {
            return zg.fail(.token(tok), "number '{s}' has leading zero", .{bytes}, &.{});
        } else {
            return zg.fail(.token(tok), "number '{s}' has leading zero", .{bytes}, &.{
                try zg.errNote(.token(tok), "use '0o' prefix for octal literals", .{}),
            });
        },
        .digit_after_base => return zg.fail(.token(tok), "expected a digit after base prefix", .{}, &.{}),
        .upper_case_base => |i| return zg.fail(.tokOff(tok, @intCast(i)), "base prefix must be lowercase", .{}, &.{}),
        .invalid_float_base => |i| return zg.fail(.tokOff(tok, @intCast(i)), "invalid base for float literal", .{}, &.{}),
        .repeated_underscore => |i| return zg.fail(.tokOff(tok, @intCast(i)), "repeated digit separator", .{}, &.{}),
        .invalid_underscore_after_special => |i| return zg.fail(.tokOff(tok, @intCast(i)), "expected digit before digit separator", .{}, &.{}),
        .invalid_digit => |info| return zg.fail(.tokOff(tok, @intCast(info.i)), "invalid digit '{c}' for {s} base", .{ bytes[info.i], @tagName(info.base) }, &.{}),
        .invalid_digit_exponent => |i| return zg.fail(.tokOff(tok, @intCast(i)), "invalid digit '{c}' in exponent", .{bytes[i]}, &.{}),
        .duplicate_exponent => |i| return zg.fail(.tokOff(tok, @intCast(i)), "duplicate exponent", .{}, &.{}),
        .exponent_after_underscore => |i| return zg.fail(.tokOff(tok, @intCast(i)), "expected digit before exponent", .{}, &.{}),
        .special_after_underscore => |i| return zg.fail(.tokOff(tok, @intCast(i)), "expected digit before '{c}'", .{bytes[i]}, &.{}),
        .trailing_special => |i| return zg.fail(.tokOff(tok, @intCast(i)), "expected digit after '{c}'", .{bytes[i - 1]}, &.{}),
        .trailing_underscore => |i| return zg.fail(.tokOff(tok, @intCast(i)), "trailing digit separator", .{}, &.{}),
        .duplicate_period => unreachable, // Validated by tokenizer
        .invalid_character => unreachable, // Validated by tokenizer
        .invalid_exponent_sign => |i| {
            assert(bytes.len >= 2 and bytes[0] == '0' and bytes[1] == 'x'); // Validated by tokenizer
            return zg.fail(.tokOff(tok, @intCast(i)), "sign '{c}' cannot follow digit '{c}' in hex base", .{ bytes[i], bytes[i - 1] }, &.{});
        },
        .period_after_exponent => |i| return zg.fail(.tokOff(tok, @intCast(i)), "unexpected period after exponent", .{}, &.{}),
    }
}

const Scope = union(enum) {
    ///// A namespace, arising from a container declaration (struct, enum, union, opaque).
    namespace: struct {
        members: *const std.AutoArrayHashMapUnmanaged(Zir.NullTerminatedString, Ast.Node.Index),
    },

    break_target: struct {
        label: ?Label,
        allow_unlabeled: bool,
        block_inst: Zir.Inst.Index,
        ri: ResultInfo,
    },
    //break_continue_target: struct {
    //    allow_unlabeled: bool,
    //    label_tok: ?Ast.TokenIndex,
    //    break_target: ?Zir.Inst.Index,
    //    continue_target: ?Zir.Inst.Index,
    //},

    ///// A `defer` statement which should be emitted when exiting past this scope.
    //@"defer": struct {
    //},

    ///// An `errdefer` statement which should be emitted when exiting past this scope.
    //@"errdefer": struct {
    //},

    /// A local variable which, due to being a simple `const`, does not have a stack allocation
    /// instruction, instead being referenced by-value.
    local_val: struct {
        name: Zir.NullTerminatedString,
        val: Zir.Ref,
        /// If `val` corresponds to the value `x`, then `ptr_strat` determines how to get the
        /// value `&x`. Rather than accessing this directly, consider using the `ptr` method.
        ptr_strat: union(enum) {
            /// The pointer exists at this `Zir.Ref`.
            ref: Zir.Ref,
            /// The pointer should be created by adding a `ref` inst at the use site.
            /// Usually, this is not okay, because the language spec requires that `&x == &x`
            /// for all local variables `x`. However, when we are in a comptime scope, Sema
            /// guarantees that `ref(x)` is the same in all cases -- either it's an equivalent
            /// comptime constant, or it's a runtime value, which is indistinguishable from
            /// other runtime values in our comptime scope.
            /// This strategy is used for function parameters when lowering generic function
            /// parameter/return types, because those type expressions are always
            /// comptime-evaluated, and it would complicate `AstAnnotate` to have it return
            /// more detailed information here.
            make_ref_inst,
            /// `AstAnnotate` told us that this variable was never evaluated as an lvalue.
            /// If it was wrong, this is a bug!
            unused,
        },

        /// If not `null`, this should be set to `true` when this local is
        /// used or discarded.
        used_or_discarded_ptr: ?*bool = null,
        used: Ast.OptionalTokenIndex = .none,
        discarded: Ast.OptionalTokenIndex = .none,

        name_token: Ast.TokenIndex,
        id_cat: IdCat,

        fn ptr(lv: @This(), zg: *ZirGen) Allocator.Error!Zir.Ref {
            switch (lv.ptr_strat) {
                .ref => |x| return x,
                .make_ref_inst => {
                    const ptr_inst = try zg.addExtended(.ref, 0, @intFromEnum(lv.val));
                    return ptr_inst.toRef();
                },
                .unused => {
                    // `ZirGen` disagrees with `AstAnnotate` on whether this local constant was
                    // evaluated as an lvalue. The bug is probably in `AstAnnotate`!
                    unreachable;
                },
            }
        }
    },
    //local_ptr: struct {
    //},

    /// These tag names are user-visible in error messages.
    const IdCat = enum {
        @"function parameter",
        @"local constant",
        @"local variable",
        @"switch tag capture",
        capture,
    };

    const Label = struct {
        tok: Ast.TokenIndex,
        used: bool,
    };
};

fn extraToU32(x: anytype) u32 {
    return switch (@typeInfo(@TypeOf(x))) {
        .int => x,
        .@"enum" => @intFromEnum(x),
        .@"struct" => @bitCast(x),
        else => comptime unreachable,
    };
}

fn u32ToExtra(comptime T: type, x: u32) T {
    return switch (@typeInfo(T)) {
        .int => x,
        .@"enum" => @enumFromInt(x),
        .@"struct" => @bitCast(x),
        else => comptime unreachable,
    };
}

fn getString(zg: *ZirGen, nts: Zir.NullTerminatedString) [:0]const u8 {
    const overlong = zg.wip_string_bytes.items[@intFromEnum(nts)..];
    const len = mem.indexOfScalar(u8, overlong, 0).?;
    return overlong[0..len :0];
}

fn lowerAstErrors(zg: *ZirGen) Allocator.Error!void {
    const gpa = zg.gpa;
    const tree = zg.tree;
    assert(tree.errors.len > 0);

    var msg: std.ArrayListUnmanaged(u8) = .empty;
    defer msg.deinit(gpa);

    var notes: std.ArrayListUnmanaged(ErrorNoteIndex) = .empty;
    defer notes.deinit(gpa);

    const token_starts = tree.tokens.items(.start);
    const token_tags = tree.tokens.items(.tag);
    const parse_err = tree.errors[0];
    const tok = parse_err.token + @intFromBool(parse_err.token_is_prev);
    const tok_start = token_starts[tok];
    const start_char = tree.source[tok_start];

    if (token_tags[tok] == .invalid and
        (start_char == '\"' or start_char == '\'' or start_char == '/' or mem.startsWith(u8, tree.source[tok_start..], "\\\\")))
    {
        const tok_len: u32 = @intCast(tree.tokenSlice(tok).len);
        const tok_end = tok_start + tok_len;
        const bad_off = blk: {
            var idx = tok_start;
            while (idx < tok_end) : (idx += 1) {
                switch (tree.source[idx]) {
                    0x00...0x09, 0x0b...0x1f, 0x7f => break,
                    else => {},
                }
            }
            break :blk idx - tok_start;
        };

        const err: Ast.Error = .{
            .tag = Ast.Error.Tag.invalid_byte,
            .token = tok,
            .extra = .{ .offset = bad_off },
        };
        msg.clearRetainingCapacity();
        try tree.renderError(err, msg.writer(gpa));
        return zg.addError(.{ .token_and_offset = .{
            .token = tok,
            .byte_offset = bad_off,
        } }, "{s}", .{msg.items}, &.{});
    }

    var cur_err = tree.errors[0];
    for (tree.errors[1..]) |err| {
        if (err.is_note) {
            try tree.renderError(err, msg.writer(gpa));
            try notes.append(gpa, try zg.errNote(.token(err.token), "{s}", .{msg.items}));
        } else {
            // Flush error
            const extra_offset = tree.errorOffset(cur_err);
            try tree.renderError(cur_err, msg.writer(gpa));
            try zg.addError(.{ .token_and_offset = .{
                .token = cur_err.token,
                .byte_offset = extra_offset,
            } }, "{s}", .{msg.items}, notes.items);
            notes.clearRetainingCapacity();
            cur_err = err;

            // TODO: `Parse` currently does not have good error recovery mechanisms, so the remaining errors could be bogus.
            // As such, we'll ignore all remaining errors for now. We should improve `Parse` so that we can report all the errors.
            return;
        }
        msg.clearRetainingCapacity();
    }

    // Flush error
    const extra_offset = tree.errorOffset(cur_err);
    try tree.renderError(cur_err, msg.writer(gpa));
    try zg.addError(.{ .token_and_offset = .{
        .token = cur_err.token,
        .byte_offset = extra_offset,
    } }, "{s}", .{msg.items}, notes.items);
}

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ast = std.zig.Ast;
const AstAnnotate = @import("AstAnnotate.zig");
const BuiltinFn = std.zig.BuiltinFn;
const assert = std.debug.assert;
const mem = std.mem;
const Zir = @import("Zir.zig");
const ZirGen = @This();
