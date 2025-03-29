//! `AstAnnotate` is a simple pass which runs over the AST before `ZirGen` to
//! determine two key pieces of information:
//!
//! * Which local constants and parameters are evaluated as lvalues
//! * Which expressions must forward result pointers to sub-expressions
//!
//! The first of the above is fairly simple. In many cases, `const` locals and
//! function parameters won't have an associated stack alloc. Since they need to
//! have a fixed address, `ZirGen` wants to know if the address of such variables
//! is ever needed so that it can create a corresponding `ref` instruction if so.
//!
//! The second requires a little more explanation.
//!
//! In some cases, `ZirGen` can choose whether to provide a result pointer or to
//! just use standard `break` instructions from a block. The latter choice can
//! result in more efficient ZIR and runtime code, but does not allow for RLS to
//! occur. Thus, we want to provide a real result pointer (from an alloc) only
//! when necessary.
//!
//! To achieve this, we need to determine which expressions require a result
//! pointer. `AstAnnotate` is responsible for locating sub-expressions which
//! consume result pointers non-trivially (e.g. writing through field pointers)
//! and recursively marking parent nodes as requiring a result location.

gpa: Allocator,
arena: Allocator,
tree: *const Ast,

wip: Result,

break_continue_targets: std.ArrayListUnmanaged(*BreakContinueTarget),
local_consts: std.ArrayListUnmanaged(LocalConst),

pub const Result = struct {
    /// Certain nodes are placed in this set under the following conditions:
    /// * if-else: either branch consumes the result pointer
    /// * labeled block: any break consumes the result pointer
    /// * switch: any prong consumes the result pointer
    /// * orelse/catch: the RHS expression consumes the result pointer
    /// * while/for: any break consumes the result pointer
    /// * @as: the second operand consumes the result pointer
    /// * const: the init expression consumes the result pointer
    /// * return: the return expression consumes the result pointer
    nodes_need_rl: NodeSet,

    /// For local `const`s and function parameters, this tracks whether they are ever evaluated as
    /// lvalues. In these cases, `ZirGen` will need to emit `ref`s for them.
    /// Because not all function parameters have an associated source node, this is based on tokens.
    /// For `const` declarations, it is the name token.
    /// For parameters, it is the name token (`Ast.full.FnProto.Param.name_token`).
    consts_need_ref: TokenSet,

    pub fn deinit(res: *Result, gpa: Allocator) void {
        res.nodes_need_rl.deinit(gpa);
        res.consts_need_ref.deinit(gpa);
    }

    pub const NodeSet = AutoHashMapUnmanaged(Ast.Node.Index, void);
    pub const TokenSet = AutoHashMapUnmanaged(Ast.TokenIndex, void);
};

fn breakWithOperand(ann: *AstAnnotate, label: ?[]const u8, operand_node: Ast.Node.Index) Allocator.Error!void {
    const bct = for (ann.break_continue_targets.items) |bct| {
        if (label) |l| {
            if (bct.label) |bl| {
                if (mem.eql(u8, l, bl)) break bct;
            }
        } else {
            if (bct.allow_unlabeled) break bct;
        }
    } else {
        // The label wasn't found; AstGen will emit an error. Just annotate the sub-expression without any result info.
        _ = try ann.expr(operand_node, .rvalue);
        return;
    };
    const res = try ann.expr(operand_node, bct.break_operand_ri);
    if (res.consumes_res_ptr) bct.consumes_break_res_ptr = true;
}
fn continueWithOperand(ann: *AstAnnotate, label: ?[]const u8, operand_node: Ast.Node.Index) Allocator.Error!void {
    const bct = for (ann.break_continue_targets.items) |bct| {
        if (label) |l| {
            if (bct.label) |bl| {
                if (mem.eql(u8, l, bl)) break bct;
            }
        } else {
            if (bct.allow_unlabeled) break bct;
        }
    } else {
        // The label wasn't found; AstGen will emit an error. Just annotate the sub-expression without any result info.
        _ = try ann.expr(operand_node, .rvalue);
        return;
    };
    const res = try ann.expr(operand_node, bct.continue_operand_ri);
    if (res.consumes_res_ptr) bct.consumes_continue_res_ptr = true;
}
fn identifierAsLvalue(ann: *AstAnnotate, ident: []const u8) Allocator.Error!void {
    for (ann.local_consts.items) |*lc| {
        if (mem.eql(u8, ident, lc.name)) {
            if (!lc.used_as_lvalue) {
                lc.used_as_lvalue = true;
                try ann.wip.consts_need_ref.putNoClobber(ann.gpa, lc.token, {});
            }
            return;
        }
    }
}

const BreakContinueTarget = struct {
    label: ?[]const u8,
    allow_unlabeled: bool,
    break_operand_ri: ResultInfo,
    continue_operand_ri: ResultInfo,
    consumes_break_res_ptr: bool = false,
    consumes_continue_res_ptr: bool = false,
};
const LocalConst = struct {
    name: []const u8,
    used_as_lvalue: bool = false,
    token: Ast.TokenIndex,
};

const ResultInfo = struct {
    eval_as_lvalue: bool,
    have_ptr: bool,
    const rvalue: ResultInfo = .{
        .eval_as_lvalue = false,
        .have_ptr = false,
    };
    const rvalue_ptr: ResultInfo = .{
        .eval_as_lvalue = false,
        .have_ptr = true,
    };
    const lvalue: ResultInfo = .{
        .eval_as_lvalue = true,
        .have_ptr = false,
    };
};
const ExprResult = struct {
    consumes_res_ptr: bool,
    const simple: ExprResult = .{ .consumes_res_ptr = false };
};

pub fn annotate(gpa: Allocator, tree: *const Ast) Allocator.Error!Result {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    var ann: AstAnnotate = .{
        .gpa = gpa,
        .arena = arena.allocator(),
        .tree = tree,
        .wip = .{
            .nodes_need_rl = .empty,
            .consts_need_ref = .empty,
        },
        .break_continue_targets = .empty,
        .local_consts = .empty,
    };
    defer {
        ann.wip.deinit(gpa);
        ann.break_continue_targets.deinit(gpa);
        ann.local_consts.deinit(gpa);
    }

    // We can't perform analysis on a broken AST. ZirGen will not run in this case.
    if (tree.errors.len == 0) {
        try ann.containerDecl(tree.containerDeclRoot());
    }

    return .{
        .nodes_need_rl = ann.wip.nodes_need_rl.move(),
        .consts_need_ref = ann.wip.consts_need_ref.move(),
    };
}

fn containerDecl(ann: *AstAnnotate, full: Ast.full.ContainerDecl) !void {
    if (full.ast.arg.unwrap()) |arg| {
        _ = try ann.expr(arg, .rvalue);
    }
    for (full.ast.members) |member_node| {
        _ = try ann.expr(member_node, .rvalue);
    }
}

fn expr(ann: *AstAnnotate, node: Ast.Node.Index, ri: ResultInfo) Allocator.Error!ExprResult {
    const tree = ann.tree;
    switch (tree.nodeTag(node)) {
        .root,
        .switch_case_one,
        .switch_case_inline_one,
        .switch_case,
        .switch_case_inline,
        .switch_range,
        .for_range,
        .asm_output,
        .asm_input,
        => unreachable,

        .@"errdefer" => {
            _ = try ann.expr(tree.nodeData(node).opt_token_and_node[1], .rvalue);
            return .simple;
        },
        .@"defer" => {
            _ = try ann.expr(tree.nodeData(node).node, .rvalue);
            return .simple;
        },

        .container_field_init,
        .container_field_align,
        .container_field,
        => {
            const full = tree.fullContainerField(node).?;
            _ = try ann.expr(full.ast.type_expr.unwrap().?, .rvalue);
            if (full.ast.align_expr.unwrap()) |e| _ = try ann.expr(e, .rvalue);
            if (full.ast.value_expr.unwrap()) |e| _ = try ann.expr(e, .rvalue);
            return .simple;
        },
        .@"usingnamespace" => {
            _ = try ann.expr(tree.nodeData(node).node, .rvalue);
            return .simple;
        },
        .test_decl => {
            _ = try ann.expr(tree.nodeData(node).opt_token_and_node[1], .rvalue);
            return .simple;
        },
        .global_var_decl,
        .local_var_decl,
        .simple_var_decl,
        .aligned_var_decl,
        => {
            const full = tree.fullVarDecl(node).?;
            const init_ri: ResultInfo = if (full.ast.type_node.unwrap()) |type_node| init_ri: {
                _ = try ann.expr(type_node, .rvalue);
                break :init_ri .rvalue_ptr;
            } else .rvalue;
            if (full.ast.align_node.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            if (full.ast.addrspace_node.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            if (full.ast.section_node.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            const init_node = full.ast.init_node.unwrap() orelse {
                // No init node, so we're done.
                return .simple;
            };
            switch (tree.tokenTag(full.ast.mut_token)) {
                .keyword_const => {
                    const name_tok = full.ast.mut_token + 1;
                    try ann.local_consts.append(ann.gpa, .{
                        .name = try ann.identString(name_tok),
                        .token = full.ast.mut_token + 1,
                    });
                    const init_res = try ann.expr(init_node, init_ri);
                    if (init_res.consumes_res_ptr) try ann.wip.nodes_need_rl.putNoClobber(ann.gpa, node, {});
                    return .simple;
                },
                .keyword_var => {
                    // We'll create an alloc either way, so don't care if the result pointer is consumed.
                    _ = try ann.expr(init_node, init_ri);
                    return .simple;
                },
                else => unreachable,
            }
        },
        .assign_destructure => {
            const full = tree.assignDestructure(node);
            for (full.ast.variables) |variable_node| {
                _ = try ann.expr(variable_node, .lvalue);
            }
            _ = try ann.expr(full.ast.value_expr, .rvalue_ptr);
            return .simple;
        },
        .assign => {
            const lhs, const rhs = tree.nodeData(node).node_and_node;
            _ = try ann.expr(lhs, .lvalue);
            _ = try ann.expr(rhs, .rvalue_ptr);
            return .simple;
        },
        .assign_shl,
        .assign_shr,
        .assign_bit_and,
        .assign_bit_or,
        .assign_bit_xor,
        .assign_div,
        .assign_sub,
        .assign_sub_wrap,
        .assign_sub_sat,
        .assign_mod,
        .assign_add,
        .assign_add_wrap,
        .assign_add_sat,
        .assign_mul,
        .assign_mul_wrap,
        .assign_mul_sat,
        => {
            const lhs, const rhs = tree.nodeData(node).node_and_node;
            _ = try ann.expr(lhs, .lvalue);
            _ = try ann.expr(rhs, .rvalue);
            return .simple;
        },
        .assign_shl_sat => {
            const lhs, const rhs = tree.nodeData(node).node_and_node;
            _ = try ann.expr(lhs, .lvalue);
            _ = try ann.expr(rhs, .rvalue);
            return .simple;
        },
        .shl, .shr => {
            const lhs, const rhs = tree.nodeData(node).node_and_node;
            _ = try ann.expr(lhs, .rvalue);
            _ = try ann.expr(rhs, .rvalue);
            return .simple;
        },
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
        .mod,
        .shl_sat,
        .bit_and,
        .bit_or,
        .bit_xor,
        .bang_equal,
        .equal_equal,
        .greater_than,
        .greater_or_equal,
        .less_than,
        .less_or_equal,
        .array_cat,
        => {
            const lhs, const rhs = tree.nodeData(node).node_and_node;
            _ = try ann.expr(lhs, .rvalue);
            _ = try ann.expr(rhs, .rvalue);
            return .simple;
        },

        .array_mult => {
            const lhs, const rhs = tree.nodeData(node).node_and_node;
            _ = try ann.expr(lhs, .rvalue);
            _ = try ann.expr(rhs, .rvalue);
            return .simple;
        },
        .error_union, .merge_error_sets => {
            const lhs, const rhs = tree.nodeData(node).node_and_node;
            _ = try ann.expr(lhs, .rvalue);
            _ = try ann.expr(rhs, .rvalue);
            return .simple;
        },
        .bool_and,
        .bool_or,
        => {
            const lhs, const rhs = tree.nodeData(node).node_and_node;
            _ = try ann.expr(lhs, .rvalue);
            _ = try ann.expr(rhs, .rvalue);
            return .simple;
        },
        .bool_not => {
            _ = try ann.expr(tree.nodeData(node).node, .rvalue);
            return .simple;
        },
        .bit_not, .negation, .negation_wrap => {
            _ = try ann.expr(tree.nodeData(node).node, .rvalue);
            return .simple;
        },

        .identifier => {
            if (ri.eval_as_lvalue) {
                try ann.identifierAsLvalue(try ann.identString(tree.nodeMainToken(node)));
            }
            return .simple;
        },

        // These nodes are leaves and never consume a result location.
        .string_literal,
        .multiline_string_literal,
        .number_literal,
        .unreachable_literal,
        .enum_literal,
        .error_value,
        .anyframe_literal,
        .char_literal,
        .error_set_decl,
        => return .simple,

        .builtin_call_two,
        .builtin_call_two_comma,
        .builtin_call,
        .builtin_call_comma,
        => {
            var buf: [2]Ast.Node.Index = undefined;
            const params = tree.builtinCallParams(&buf, node).?;
            return ann.builtinCall(ri, node, params);
        },

        .call_one,
        .call_one_comma,
        .async_call_one,
        .async_call_one_comma,
        .call,
        .call_comma,
        .async_call,
        .async_call_comma,
        => {
            var buf: [1]Ast.Node.Index = undefined;
            const full = tree.fullCall(&buf, node).?;
            if (tree.nodeTag(full.ast.fn_expr) == .field_access) {
                // Method call syntax; evaluate the LHS as an lvalue.
                const lhs, _ = tree.nodeData(node).node_and_token;
                _ = try ann.expr(lhs, .lvalue);
            } else {
                // Normal call; evaluate the function as an rvalue.
                _ = try ann.expr(full.ast.fn_expr, .rvalue);
            }
            for (full.ast.params) |param_node| {
                _ = try ann.expr(param_node, .rvalue);
            }
            return switch (tree.nodeTag(node)) {
                .call_one,
                .call_one_comma,
                .call,
                .call_comma,
                => .simple, // TODO: once function calls are passed result locations this will change
                .async_call_one,
                .async_call_one_comma,
                .async_call,
                .async_call_comma,
                => .{ .consumes_res_ptr = ri.have_ptr }, // always use result ptr for frames
                else => unreachable,
            };
        },

        .@"return" => {
            if (tree.nodeData(node).opt_node.unwrap()) |lhs| {
                const res = try ann.expr(lhs, .rvalue_ptr);
                if (res.consumes_res_ptr) {
                    try ann.wip.nodes_need_rl.putNoClobber(ann.gpa, node, {});
                }
            }
            return .simple;
        },

        .field_access => {
            const lhs, _ = tree.nodeData(node).node_and_token;
            _ = try ann.expr(lhs, .lvalue); // always lvalue as optimization
            return .simple;
        },

        .if_simple, .@"if" => {
            const full = tree.fullIf(node).?;
            if (full.error_token != null or full.payload_token != null) {
                const payload_is_ref = if (full.payload_token) |t| tree.tokenTag(t) == .asterisk else false;
                _ = try ann.expr(full.ast.cond_expr, if (payload_is_ref) .lvalue else .rvalue);
            } else {
                _ = try ann.expr(full.ast.cond_expr, .rvalue);
            }

            const then_res = try ann.expr(full.ast.then_expr, ri);
            const else_res: ExprResult = if (full.ast.else_expr.unwrap()) |else_expr| r: {
                break :r try ann.expr(else_expr, ri);
            } else .{ .consumes_res_ptr = false };
            const uses_rl = then_res.consumes_res_ptr or else_res.consumes_res_ptr;
            if (uses_rl) try ann.wip.nodes_need_rl.putNoClobber(ann.gpa, node, {});
            return .{ .consumes_res_ptr = uses_rl };
        },

        .while_simple, .while_cont, .@"while" => {
            const full = tree.fullWhile(node).?;
            const label: ?[]const u8 = if (full.label_token) |label_token| label: {
                break :label try ann.identString(label_token);
            } else null;
            if (full.error_token != null or full.payload_token != null) {
                const payload_is_ref = if (full.payload_token) |t| tree.tokenTag(t) == .asterisk else false;
                _ = try ann.expr(full.ast.cond_expr, if (payload_is_ref) .lvalue else .rvalue);
            } else {
                _ = try ann.expr(full.ast.cond_expr, .rvalue);
            }
            var bct: BreakContinueTarget = .{
                .label = label,
                .allow_unlabeled = true,
                .break_operand_ri = ri,
                .continue_operand_ri = .rvalue,
            };
            try ann.break_continue_targets.append(ann.gpa, &bct);
            defer assert(ann.break_continue_targets.pop().? == &bct);
            if (full.ast.cont_expr.unwrap()) |cont_expr| _ = try ann.expr(cont_expr, .rvalue);
            _ = try ann.expr(full.ast.then_expr, .rvalue);
            const else_res: ExprResult = if (full.ast.else_expr.unwrap()) |else_expr| r: {
                break :r try ann.expr(else_expr, ri);
            } else .{ .consumes_res_ptr = false };
            if (bct.consumes_break_res_ptr or else_res.consumes_res_ptr) {
                try ann.wip.nodes_need_rl.putNoClobber(ann.gpa, node, {});
                return .{ .consumes_res_ptr = true };
            } else {
                return .{ .consumes_res_ptr = false };
            }
        },

        .for_simple, .@"for" => {
            const full = tree.fullFor(node).?;
            const label: ?[]const u8 = if (full.label_token) |label_token| label: {
                break :label try ann.identString(label_token);
            } else null;
            for (full.ast.inputs) |input| {
                if (tree.nodeTag(input) == .for_range) {
                    const lhs, const opt_rhs = tree.nodeData(input).node_and_opt_node;
                    _ = try ann.expr(lhs, .rvalue);
                    if (opt_rhs.unwrap()) |rhs| _ = try ann.expr(rhs, .rvalue);
                } else {
                    _ = try ann.expr(input, .rvalue);
                }
            }
            var bct: BreakContinueTarget = .{
                .label = label,
                .allow_unlabeled = true,
                .break_operand_ri = ri,
                .continue_operand_ri = .rvalue,
            };
            try ann.break_continue_targets.append(ann.gpa, &bct);
            defer assert(ann.break_continue_targets.pop().? == &bct);
            _ = try ann.expr(full.ast.then_expr, .rvalue);
            const else_res: ExprResult = if (full.ast.else_expr.unwrap()) |else_expr| r: {
                break :r try ann.expr(else_expr, ri);
            } else .{ .consumes_res_ptr = false };
            if (bct.consumes_break_res_ptr or else_res.consumes_res_ptr) {
                try ann.wip.nodes_need_rl.putNoClobber(ann.gpa, node, {});
                return .{ .consumes_res_ptr = true };
            } else {
                return .{ .consumes_res_ptr = false };
            }
        },

        .slice_open, .slice, .slice_sentinel => {
            const full = tree.fullSlice(node).?;
            _ = try ann.expr(full.ast.sliced, .lvalue);
            _ = try ann.expr(full.ast.start, .rvalue);
            if (full.ast.end.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            if (full.ast.sentinel.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            return .simple;
        },
        .deref => {
            _ = try ann.expr(tree.nodeData(node).node, .rvalue);
            return .simple;
        },
        .address_of => {
            _ = try ann.expr(tree.nodeData(node).node, .lvalue);
            return .simple;
        },
        .optional_type => {
            _ = try ann.expr(tree.nodeData(node).node, .rvalue);
            return .simple;
        },
        .@"nosuspend" => return ann.expr(tree.nodeData(node).node, ri),
        .@"try" => return ann.expr(tree.nodeData(node).node, .{
            .eval_as_lvalue = ri.eval_as_lvalue,
            .have_ptr = false,
        }),
        .grouped_expression => return ann.expr(tree.nodeData(node).node_and_token[0], ri),
        .unwrap_optional => return ann.expr(tree.nodeData(node).node_and_token[0], .{
            .eval_as_lvalue = ri.eval_as_lvalue,
            .have_ptr = false,
        }),

        .block_two,
        .block_two_semicolon,
        .block,
        .block_semicolon,
        => {
            var buf: [2]Ast.Node.Index = undefined;
            const statements = tree.blockStatements(&buf, node).?;

            const lbrace = tree.nodeMainToken(node);
            const label: ?[]const u8 = if (tree.isTokenPrecededByTags(lbrace, &.{ .identifier, .colon })) l: {
                break :l try ann.identString(lbrace - 2);
            } else null;

            // The block will put local consts in scope; we'll delete them once the body is done.
            const old_local_consts_len = ann.local_consts.items.len;
            defer ann.local_consts.shrinkRetainingCapacity(old_local_consts_len);

            var bct: BreakContinueTarget = .{
                .label = label,
                .allow_unlabeled = false,
                .break_operand_ri = ri,
                .continue_operand_ri = .rvalue,
            };
            try ann.break_continue_targets.append(ann.gpa, &bct);
            defer assert(ann.break_continue_targets.pop().? == &bct);

            for (statements) |stmt| {
                _ = try ann.expr(stmt, .rvalue);
            }

            return .{ .consumes_res_ptr = bct.consumes_break_res_ptr };
        },
        .anyframe_type => {
            _, const child_type = tree.nodeData(node).token_and_node;
            _ = try ann.expr(child_type, .rvalue);
            return .simple;
        },
        .@"catch", .@"orelse" => {
            const lhs, const rhs = tree.nodeData(node).node_and_node;
            _ = try ann.expr(lhs, .{
                .eval_as_lvalue = ri.eval_as_lvalue,
                .have_ptr = false,
            });
            const rhs_res = try ann.expr(rhs, ri);
            if (rhs_res.consumes_res_ptr) try ann.wip.nodes_need_rl.putNoClobber(ann.gpa, node, {});
            return .{ .consumes_res_ptr = rhs_res.consumes_res_ptr };
        },

        .ptr_type_aligned,
        .ptr_type_sentinel,
        .ptr_type,
        .ptr_type_bit_range,
        => {
            const full = tree.fullPtrType(node).?;
            _ = try ann.expr(full.ast.child_type, .rvalue);
            if (full.ast.sentinel.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            if (full.ast.addrspace_node.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            if (full.ast.align_node.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            if (full.ast.bit_range_start.unwrap()) |bit_range_start| {
                const bit_range_end = full.ast.bit_range_end.unwrap().?;
                _ = try ann.expr(bit_range_start, .rvalue);
                _ = try ann.expr(bit_range_end, .rvalue);
            }
            return .simple;
        },

        .container_decl,
        .container_decl_trailing,
        .container_decl_arg,
        .container_decl_arg_trailing,
        .container_decl_two,
        .container_decl_two_trailing,
        .tagged_union,
        .tagged_union_trailing,
        .tagged_union_enum_tag,
        .tagged_union_enum_tag_trailing,
        .tagged_union_two,
        .tagged_union_two_trailing,
        => {
            var buf: [2]Ast.Node.Index = undefined;
            try ann.containerDecl(tree.fullContainerDecl(&buf, node).?);
            return .simple;
        },

        .@"break" => {
            const opt_label, const opt_rhs = tree.nodeData(node).opt_token_and_opt_node;
            const rhs = opt_rhs.unwrap() orelse {
                // `break` with void is not interesting
                return .simple;
            };
            const label_str: ?[]const u8 = if (opt_label.unwrap()) |label_token| str: {
                break :str try ann.identString(label_token);
            } else null;
            try ann.breakWithOperand(label_str, rhs);
            return .simple;
        },

        .@"continue" => {
            const opt_label, const opt_rhs = tree.nodeData(node).opt_token_and_opt_node;
            const rhs = opt_rhs.unwrap() orelse {
                // `continue` with void is not interesting
                return .simple;
            };
            const label_str: ?[]const u8 = if (opt_label.unwrap()) |label_token| str: {
                break :str try ann.identString(label_token);
            } else null;
            try ann.continueWithOperand(label_str, rhs);
            return .simple;
        },

        .asm_simple, .@"asm" => {
            const full = tree.fullAsm(node).?;
            for (full.outputs) |n| _ = try ann.expr(n, .lvalue);
            for (full.inputs) |n| _ = try ann.expr(n, .rvalue);
            return .simple;
        },

        .array_type, .array_type_sentinel => {
            const full = tree.fullArrayType(node).?;
            _ = try ann.expr(full.ast.elem_count, .rvalue);
            _ = try ann.expr(full.ast.elem_type, .rvalue);
            if (full.ast.sentinel.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            return .simple;
        },
        .array_access => {
            const lhs, const rhs = tree.nodeData(node).node_and_node;
            _ = try ann.expr(lhs, .lvalue); // always lvalue as optimization
            _ = try ann.expr(rhs, .rvalue);
            return .simple;
        },
        .@"comptime" => {
            // ZirGen will emit an error if the scope is already comptime, so we can assume it is
            // not. This means the result location is not forwarded.
            _ = try ann.expr(tree.nodeData(node).node, .{
                .eval_as_lvalue = ri.eval_as_lvalue,
                .have_ptr = false,
            });
            return .simple;
        },
        .@"switch", .switch_comma => {
            const full = tree.fullSwitch(node).?;

            const cond_as_lvalue = for (full.ast.cases) |case_node| {
                if (tree.fullSwitchCase(case_node).?.payload_token) |t| {
                    if (tree.tokenTag(t) == .asterisk) {
                        // There's a by-ref capture, so we evaluate the condition as an lvalue.
                        break true;
                    }
                }
            } else false;

            _ = try ann.expr(full.ast.condition, .{
                .eval_as_lvalue = cond_as_lvalue,
                .have_ptr = false,
            });

            var bct: BreakContinueTarget = .{
                .label = if (full.label_token) |t| try ann.identString(t) else null,
                .allow_unlabeled = false,
                .break_operand_ri = ri,
                .continue_operand_ri = .{
                    .eval_as_lvalue = cond_as_lvalue,
                    .have_ptr = false,
                },
            };
            try ann.break_continue_targets.append(ann.gpa, &bct);
            defer assert(ann.break_continue_targets.pop().? == &bct);

            var any_prong_directly_consumes_rl = false;
            for (full.ast.cases) |case_node| {
                const case = tree.fullSwitchCase(case_node).?;
                for (case.ast.values) |item_node| {
                    if (tree.nodeTag(item_node) == .switch_range) {
                        const lhs, const rhs = tree.nodeData(item_node).node_and_node;
                        _ = try ann.expr(lhs, .rvalue);
                        _ = try ann.expr(rhs, .rvalue);
                    } else {
                        _ = try ann.expr(item_node, .rvalue);
                    }
                }
                const res = try ann.expr(case.ast.target_expr, ri);
                if (res.consumes_res_ptr) {
                    any_prong_directly_consumes_rl = true;
                }
            }
            const rl_consumed = any_prong_directly_consumes_rl or (full.label_token != null and bct.consumes_break_res_ptr);
            if (rl_consumed) try ann.wip.nodes_need_rl.putNoClobber(ann.gpa, node, {});
            return .{ .consumes_res_ptr = rl_consumed };
        },
        .@"suspend" => {
            _ = try ann.expr(tree.nodeData(node).node, .rvalue);
            return .simple;
        },
        .@"await", .@"resume" => {
            _ = try ann.expr(tree.nodeData(node).node, .lvalue);
            return .simple;
        },

        .array_init_one,
        .array_init_one_comma,
        .array_init_dot_two,
        .array_init_dot_two_comma,
        .array_init_dot,
        .array_init_dot_comma,
        .array_init,
        .array_init_comma,
        => {
            var buf: [2]Ast.Node.Index = undefined;
            const full = tree.fullArrayInit(&buf, node).?;

            if (full.ast.type_expr.unwrap()) |type_expr| {
                // Explicitly typed init does not participate in RLS
                _ = try ann.expr(type_expr, .rvalue);
                for (full.ast.elements) |elem_init| {
                    _ = try ann.expr(elem_init, .rvalue);
                }
                return .simple;
            }

            // If we have a result pointer, we use and forward it
            for (full.ast.elements) |elem_init| {
                _ = try ann.expr(elem_init, .{
                    .eval_as_lvalue = false,
                    .have_ptr = ri.have_ptr,
                });
            }
            return .{ .consumes_res_ptr = ri.have_ptr };
        },

        .struct_init_one,
        .struct_init_one_comma,
        .struct_init_dot_two,
        .struct_init_dot_two_comma,
        .struct_init_dot,
        .struct_init_dot_comma,
        .struct_init,
        .struct_init_comma,
        => {
            var buf: [2]Ast.Node.Index = undefined;
            const full = tree.fullStructInit(&buf, node).?;

            if (full.ast.type_expr.unwrap()) |type_expr| {
                // Explicitly typed init does not participate in RLS
                _ = try ann.expr(type_expr, .rvalue);
                for (full.ast.fields) |field_init| {
                    _ = try ann.expr(field_init, .rvalue);
                }
                return .simple;
            }

            // If we have a result pointer, we use and forward it
            for (full.ast.fields) |field_init| {
                _ = try ann.expr(field_init, .{
                    .eval_as_lvalue = false,
                    .have_ptr = ri.have_ptr,
                });
            }
            return .{ .consumes_res_ptr = ri.have_ptr };
        },

        .fn_proto_simple,
        .fn_proto_multi,
        .fn_proto_one,
        .fn_proto,
        .fn_decl,
        => |tag| {
            var buf: [1]Ast.Node.Index = undefined;
            const full = tree.fullFnProto(&buf, node).?;
            const body_node = if (tag == .fn_decl) tree.nodeData(node).node_and_node[1].toOptional() else .none;

            // We'll add the parameters to `local_consts`; save the old length to reset it after.
            const old_local_consts_len = ann.local_consts.items.len;
            defer ann.local_consts.shrinkRetainingCapacity(old_local_consts_len);

            {
                var it = full.iterate(tree);
                while (it.next()) |param| {
                    if (param.type_expr) |n| _ = try ann.expr(n, .rvalue);
                    if (param.name_token) |t| try ann.local_consts.append(ann.gpa, .{
                        .name = try ann.identString(t),
                        .token = t,
                    });
                }
            }
            if (full.ast.align_expr.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            if (full.ast.addrspace_expr.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            if (full.ast.section_expr.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            if (full.ast.callconv_expr.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            const return_type = full.ast.return_type.unwrap().?;
            _ = try ann.expr(return_type, .rvalue);
            if (body_node.unwrap()) |n| _ = try ann.expr(n, .rvalue);
            return .simple;
        },
    }
}

fn identString(ann: *AstAnnotate, token: Ast.TokenIndex) ![]const u8 {
    const tree = ann.tree;
    assert(tree.tokenTag(token) == .identifier);
    const ident_name = tree.tokenSlice(token);
    if (!std.mem.startsWith(u8, ident_name, "@")) {
        return ident_name;
    }
    return std.zig.string_literal.parseAlloc(ann.arena, ident_name[1..]) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidLiteral => "", // This pass can safely return garbage on invalid AST
    };
}

fn builtinCall(ann: *AstAnnotate, ri: ResultInfo, node: Ast.Node.Index, args: []const Ast.Node.Index) Allocator.Error!ExprResult {
    const tree = ann.tree;
    const builtin_token = tree.nodeMainToken(node);
    const builtin_name = tree.tokenSlice(builtin_token);
    const info = BuiltinFn.list.get(builtin_name) orelse return .simple;
    if (info.param_count) |expected| {
        if (expected != args.len) return .simple;
    }
    switch (info.tag) {
        .field => {
            _ = try ann.expr(args[0], .lvalue); // always lvalue as optimization
            return .simple;
        },
        // TODO: this is a workaround for llvm/llvm-project#68409
        // Zig tracking issue: #16876
        .frame_address => return .{ .consumes_res_ptr = ri.have_ptr },
        // The following tags are builtins which:
        // * evaluate all args as rvalues without result pointers
        // * do not consume their result pointer
        // This is the vast majority of builtins!
        .branch_hint,
        .import,
        .c_import,
        .union_init,
        .bit_cast,
        .as,
        .min,
        .max,
        .compile_log,
        .TypeOf,
        .@"extern",
        .@"export",
        .src,
        .This,
        .return_address,
        .error_return_trace,
        .frame,
        .breakpoint,
        .disable_instrumentation,
        .disable_intrinsics,
        .in_comptime,
        .panic,
        .trap,
        .c_va_start,
        .size_of,
        .bit_size_of,
        .align_of,
        .compile_error,
        .set_eval_branch_quota,
        .int_from_bool,
        .int_from_error,
        .error_from_int,
        .embed_file,
        .error_name,
        .set_runtime_safety,
        .Type,
        .c_undef,
        .c_include,
        .wasm_memory_size,
        .splat,
        .set_float_mode,
        .type_info,
        .work_item_id,
        .work_group_size,
        .work_group_id,
        .int_from_ptr,
        .int_from_enum,
        .sqrt,
        .sin,
        .cos,
        .tan,
        .exp,
        .exp2,
        .log,
        .log2,
        .log10,
        .abs,
        .floor,
        .ceil,
        .trunc,
        .round,
        .tag_name,
        .type_name,
        .Frame,
        .frame_size,
        .int_from_float,
        .float_from_int,
        .ptr_from_int,
        .enum_from_int,
        .float_cast,
        .int_cast,
        .truncate,
        .error_cast,
        .ptr_cast,
        .align_cast,
        .addrspace_cast,
        .const_cast,
        .volatile_cast,
        .clz,
        .ctz,
        .pop_count,
        .byte_swap,
        .bit_reverse,
        .div_exact,
        .div_floor,
        .div_trunc,
        .mod,
        .rem,
        .shl_exact,
        .shr_exact,
        .bit_offset_of,
        .offset_of,
        .has_decl,
        .has_field,
        .FieldType,
        .field_parent_ptr,
        .wasm_memory_grow,
        .c_define,
        .reduce,
        .add_with_overflow,
        .sub_with_overflow,
        .mul_with_overflow,
        .shl_with_overflow,
        .atomic_load,
        .atomic_rmw,
        .atomic_store,
        .mul_add,
        .call,
        .memcpy,
        .memset,
        .shuffle,
        .select,
        .async_call,
        .Vector,
        .prefetch,
        .c_va_arg,
        .c_va_copy,
        .c_va_end,
        .cmpxchg_strong,
        .cmpxchg_weak,
        => {
            for (args) |arg| _ = try ann.expr(arg, .rvalue);
            return .simple;
        },
    }
}

const std = @import("std");

const AstAnnotate = @This();
const Ast = std.zig.Ast;
const BuiltinFn = std.zig.BuiltinFn;

const Allocator = std.mem.Allocator;
const AutoHashMapUnmanaged = std.AutoHashMapUnmanaged;
const assert = std.debug.assert;
const mem = std.mem;
