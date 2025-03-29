instructions: std.MultiArrayList(Inst.Repr).Slice,
extra: []u32,
limbs: []std.math.big.Limb,
string_bytes: []u8,
compile_errors: []CompileError,
error_notes: []CompileError.Note,
imports: []NullTerminatedString,

pub fn deinit(zir: *Zir, gpa: Allocator) void {
    zir.instructions.deinit(gpa);
    gpa.free(zir.extra);
    gpa.free(zir.limbs);
    gpa.free(zir.string_bytes);
    gpa.free(zir.compile_errors);
    gpa.free(zir.error_notes);
    gpa.free(zir.imports);
}

pub const CompileError = extern struct {
    msg: NullTerminatedString,
    src: Src.Repr,
    /// Ignored if `note_count == 0`.
    first_note: u32,
    note_count: u32,

    pub fn getNotes(err: CompileError, zir: *const Zir) []const Note {
        return zir.error_notes[err.first_note..][0..err.note_count];
    }

    pub const Note = extern struct {
        msg: NullTerminatedString,
        src: Src.Repr,
    };

    comptime {
        assert(std.meta.hasUniqueRepresentation(CompileError));
        assert(std.meta.hasUniqueRepresentation(Note));
    }

    pub const Src = union(enum) {
        token_and_offset: struct {
            token: Ast.TokenIndex,
            byte_offset: u32,
        },
        node: Ast.Node.Index,

        pub fn token(t: Ast.TokenIndex) Src {
            return .{ .token_and_offset = .{
                .token = t,
                .byte_offset = 0,
            } };
        }

        pub fn tokOff(t: Ast.TokenIndex, i: u32) Src {
            return .{ .token_and_offset = .{
                .token = t,
                .byte_offset = i,
            } };
        }

        const Repr = extern struct {
            token: Ast.OptionalTokenIndex,
            /// If `token == .none`, this is an `Ast.Node.Index`.
            /// Otherwise, this is a byte offset into `token`.
            node_or_offset: u32,

            pub fn unwrap(repr: Repr) Src {
                return if (repr.token.unwrap()) |t| .{ .token_and_offset = .{
                    .token = t,
                    .byte_offset = repr.node_or_offset,
                } } else .{ .node = @enumFromInt(repr.node_or_offset) };
            }
            pub fn init(src: Src) Repr {
                return switch (src) {
                    .token_and_offset => |tao| .{
                        .token = .fromToken(tao.token),
                        .node_or_offset = tao.byte_offset,
                    },
                    .node => |n| .{
                        .token = .none,
                        .node_or_offset = @intFromEnum(n),
                    },
                };
            }
        };
    };
};

pub const Inst = union(enum) {
    declaration: Declaration,

    int: u64,
    int_big: Limbs,

    float: f128,

    str: struct {
        index: u32,
        len: u32,
        pub fn get(str: @This(), zir: *const Zir) []const u8 {
            return zir.string_bytes[str.index..][0..str.len];
        }
    },

    ret_implicit,
    ret: Ref,

    add: BinOp,
    add_wrap: BinOp,
    add_sat: BinOp,
    sub: BinOp,
    sub_wrap: BinOp,
    sub_sat: BinOp,
    mul: BinOp,
    mul_wrap: BinOp,
    mul_sat: BinOp,
    div: BinOp,
    mod_rem: BinOp,
    shl_sat: BinOp,

    negate: Ref,

    coerce: struct {
        operand: Ref,
        dest_ty: Ref,
    },
    store: struct {
        ptr: Ref,
        operand: Ref,
    },
    load: Ref,

    import: struct {
        res_ty: Ref,
        path: NullTerminatedString,
    },

    block: struct {
        next_inst: Inst.Index,
        body: Body,
    },
    @"break": Break,
    struct_decl: struct {
        next_inst: Inst.Index,
        fields_hash: std.zig.SrcHash,
        src_line: u32,
        captures: []const TypeCapture,
        capture_names: []const NullTerminatedString,
        backing_int_body: ?Body,
        decls: Declarations,
        fields: StructFields,
    },
    decl_ref: NullTerminatedString,
    validate_const: Ref,
    validate_const_ref: Ref,
    ref: Ref,
    ensure_result_used: Ref,
    int_type: std.builtin.Type.Int,
    zirgen_error,

    pub const Repr = struct {
        tag: Tag,
        data: [2]u32,
        pub const Tag = enum(u8) {
            /// Corresponds to `Inst.int`.
            /// `data` is a bitcast `u64`.
            int,
            /// Corresponds to `Inst.float`.
            /// `data` is a bitcast `f64`.
            float64,
            /// Corresponds to `Inst.str`.
            /// `data[0]` is `index: u32`.
            /// `data[1]` is `len: u32`.
            str,

            // The following instructions are all binary operators which use `data` as follows:
            // `data[0]` is `lhs: Ref`.
            // `data[1]` is `rhs: Ref`.

            /// Corresponds to `Inst.add`.
            add,
            /// Corresponds to `Inst.add_wrap`.
            add_wrap,
            /// Corresponds to `Inst.add_sat`.
            add_sat,
            /// Corresponds to `Inst.sub`.
            sub,
            /// Corresponds to `Inst.sub_wrap`.
            sub_wrap,
            /// Corresponds to `Inst.sub_sat`.
            sub_sat,
            /// Corresponds to `Inst.mul`.
            mul,
            /// Corresponds to `Inst.mul_wrap`.
            mul_wrap,
            /// Corresponds to `Inst.mul_sat`.
            mul_sat,
            /// Corresponds to `Inst.div`.
            div,
            /// Corresponds to `Inst.mod_rem`.
            mod_rem,
            /// Corresponds to `Inst.shl_sat`.
            shl_sat,

            /// Corresponds to `Inst.coerce`.
            /// `data[0]` is `operand: Ref`.
            /// `data[1]` is `dest_ty: Ref`.
            coerce,
            /// Corresponds to `Inst.store`.
            /// `data[0]` is `ptr: Ref`.
            /// `data[1]` is `operand: Ref`.
            store,

            /// Corresponds to `Inst.import`.
            /// `data[0]` is `res_ty: Ref`.
            /// `data[1]` is `path: NullTerminatedString`.
            import,

            /// Corresponds to `Inst.block`.
            /// The instruction at the next index is the start of the block body.
            /// `data[0]` is `next_inst: Inst.Index`.
            /// `data[1]` is unused.
            block,
            /// Corresponds to `Inst.@"break"`.
            /// `data[0]` is `target: Inst.Index`.
            /// `data[1]` is `operand: Ref`.
            @"break",
            /// `data[0]` is `Extended`.
            /// `data[1]` is interpreted based on `Extended.tag`.
            extended,
        };
        pub const Extended = packed struct(u32) {
            tag: Extended.Tag,
            small: u16,
            pub const Tag = enum(u16) {
                /// Does not correspond to any tag of `Inst`.
                /// This instruction never appears in a body, and should never have `get` called on it.
                value_placeholder,
                /// Corresponds to `Inst.struct_decl`.
                /// `small` is `StructDecl.Small`.
                /// `data[1]` is extra index to `StructDecl`.
                struct_decl,
                /// Corresponds to `Inst.decl_ref`.
                /// `small` is unused and always 0.
                /// `data[1]` is `decl_name: NullTerminatedString`.
                decl_ref,
                /// Corresponds to `Inst.load`.
                /// `small` is unused and always 0.
                /// `data[1]` is `operand: Ref`.
                load,
                /// Corresponds to `Inst.negate`.
                /// `small` is unused and always 0.
                /// `data[1]` is `operand: Ref`.
                negate,
                /// Corresponds to `Inst.int_big`.
                /// `small` is the number of limbs.
                /// `data[1]` is extra index to those limbs (ptrcast appropriately to u32 slice).
                int_big,
                /// Corresponds to `Inst.float`.
                /// `small` is unused and always 0.
                /// `data[1]` is extra index to a bitcast `f128`.
                float128,
                /// Corresponds to `Inst.validate_const`.
                /// `small` is unused and always 0.
                /// `data[1]` is `operand: Ref`.
                validate_const,
                /// Corresponds to `Inst.validate_const_ref`.
                /// `small` is unused and always 0.
                /// `data[1]` is `operand: Ref`.
                validate_const_ref,
                /// Corresponds to `Inst.ref`.
                /// `small` is unused and always 0.
                /// `data[1]` is `operand: Ref`.
                ref,
                /// Corresponds to `Inst.ensure_result_used`.
                /// `small` is unused and always 0.
                /// `data[1]` is `operand: Ref`.
                ensure_result_used,
                /// Corresponds to `Inst.int_type`.
                /// `small` is `bits`.
                /// `data[1]` is `@intFromEnum(signedness)`.
                int_type,
                /// Corresponds to `Inst.zirgen_error`.
                /// `small` is unused and always 0.
                /// `data[1]` is unused and always 0.
                zirgen_error,
                /// Corresponds to `Inst.declaration` with `Declaration.kind == .@"usingnamespace"`.
                /// `small` is `@intFromBool(is_pub)`.
                /// `data[1]` is extra index to `DeclSimple`.
                decl_usingnamespace,
                /// Corresponds to `Inst.declaration` with `Declaration.kind == .@"comptime"`.
                /// `small` is unused and always 0.
                /// `data[1]` is extra index to `DeclSimple`.
                decl_comptime,
                /// Corresponds to `Inst.declaration` with `Declaration.kind == .unnamed_test`.
                /// `small` is unused and always 0.
                /// `data[1]` is extra index to `DeclSimple`.
                decl_unnamed_test,
                /// Corresponds to `Inst.declaration` with `Declaration.kind == .@"test"`.
                /// `small` is unused and always 0.
                /// `data[1]` is extra index to `DeclNamedTest`.
                decl_named_test,
                /// Corresponds to `Inst.declaration` with `Declaration.kind == .decltest`.
                /// `small` is unused and always 0.
                /// `data[1]` is extra index to `DeclNamedTest`.
                decl_decltest,
                /// Corresponds to `Inst.declaration` with `Declaration.kind == .@"const"`.
                /// `small` is `DeclConstVar.Small`.
                /// `data[1]` is extra index to `DeclConstVar`.
                decl_const,
                /// Corresponds to `Inst.declaration` with `Declaration.kind == .@"var"`.
                /// `small` is `DeclConstVar.Small`.
                /// `data[1]` is extra index to `DeclConstVar`.
                decl_var,
                /// Corresponds to `Inst.declaration` with `Declaration.kind == .@"fn"`.
                /// `small` is `DeclFn.Small`.
                /// `data[1]` is extra index to `DeclFn`.
                decl_fn,
                /// Corresponds to `Inst.ret_implicit`.
                /// `small` is unused and always 0.
                /// `data[1]` is unused and always 0.
                ret_implicit,
                /// Corresponds to `Inst.ret`.
                /// `small` is unused and always 0.
                /// `data[1]` is `operand: Ref`.
                ret,
            };

            pub const DeclSimple = struct {
                // These fields should be stored in an array and bitcast to a `std.zig.SrcHash`.
                src_hash_0: u32,
                src_hash_1: u32,
                src_hash_2: u32,
                src_hash_3: u32,

                src_line: u32,
                src_column: u32,

                body_start: Inst.Index,
            };

            pub const DeclNamedTest = struct {
                // These fields should be stored in an array and bitcast to a `std.zig.SrcHash`.
                src_hash_0: u32,
                src_hash_1: u32,
                src_hash_2: u32,
                src_hash_3: u32,

                src_line: u32,
                src_column: u32,

                /// If the extended instruction tag was `.decl_decltest`, this is the name of the declaration being tested.
                name: NullTerminatedString,

                body_start: Inst.Index,
            };

            /// Trailing data:
            /// * `value_body_start: Inst.Index`        // if `linkage != .@"extern"`
            /// * `type_body_start: Inst.Index`         // if `has_type`
            /// * `align_body_start: Inst.Index`        // if `has_align`
            /// * `linksection_body_start: Inst.Index`  // if `has_linksection`
            /// * `addrspace_body_start: Inst.Index`    // if `has_addrspace`
            /// * `lib_name: NullTerminatedString`      // if `has_lib_name`
            pub const DeclConstVar = struct {
                // These fields should be stored in an array and bitcast to a `std.zig.SrcHash`.
                src_hash_0: u32,
                src_hash_1: u32,
                src_hash_2: u32,
                src_hash_3: u32,

                src_line: u32,
                src_column: u32,

                name: NullTerminatedString,

                pub const Small = packed struct(u16) {
                    is_pub: bool,
                    is_threadlocal: bool,

                    has_type: bool,
                    has_align: bool,
                    has_linksection: bool,
                    has_addrspace: bool,

                    linkage: Declaration.Linkage,
                    has_lib_name: bool,

                    _: u7 = 0,
                };
            };

            /// Trailing:
            /// * align_body_start: Inst.Index        // if has_align
            /// * linksection_body_start: Inst.Index  // if has_linksection
            /// * addrspace_body_start: Inst.Index    // if has_addrspace
            /// * callconv_body_start: Inst.Index     // if has_callconv
            /// * lib_name: NullTerminatedString      // if has_lib_name
            /// * body_start: Inst.Index              // if linkage != .@"extern"
            /// * param: {   // for each params_len
            ///     name: NullTerminatedString
            ///     type_body_start: Inst.Index.Optional  // `.none` means `anytype`
            ///     placeholder_inst: Inst.Index
            ///   }
            /// * param_is_comptime: u32              // if any_comptime_params; for every 32 parameters; LSB is first parameter
            /// * param_is_noalias: u32               // if any_noalias_params; for every 32 parameters; LSB is first parameter
            /// * param_ty_is_generic: u32            // if any_generic_param_ty; for every 32 parameters; LSB is first parameter; false for anytype params
            pub const DeclFn = struct {
                // These fields should be stored in an array and bitcast to a `std.zig.SrcHash`.
                src_hash_0: u32,
                src_hash_1: u32,
                src_hash_2: u32,
                src_hash_3: u32,

                // These fields should be stored in an array and bitcast to a `std.zig.SrcHash`.
                proto_hash_0: u32,
                proto_hash_1: u32,
                proto_hash_2: u32,
                proto_hash_3: u32,

                src_line: u32,
                src_column: u32,

                name: NullTerminatedString,

                ret_ty_body_start: Inst.Index,
                params_len: u32,

                pub const Small = packed struct(u16) {
                    is_pub: bool,

                    has_align: bool,
                    has_linksection: bool,
                    has_addrspace: bool,
                    has_callconv: bool,

                    /// This is mutually exclusive with `has_callconv`.
                    callconv_inline: bool,

                    any_comptime_params: bool,
                    any_noalias_params: bool,
                    any_generic_param_ty: bool,
                    ret_ty_is_generic: bool,

                    is_inferred_error: bool,
                    is_var_args: bool,
                    is_noinline: bool,

                    linkage: Declaration.Linkage,
                    has_lib_name: bool,
                };
            };

            /// Trailing data:
            /// * captures_len: u32     // if `any_captures`
            /// * fields_len: u32       // if `any_fields`
            /// * decls_len: u32        // if `any_decls`
            /// * capture: TypeCapture  // for each `captures_len`
            /// * capture_name: NullTerminatedString  // for each `captures_len`
            /// * backing_int_body_start: Inst.Index  // if `has_backing_int`
            /// * field_is_comptime: u32 // if `any_comptime_fields`; for every 32 fields; LSB is first field
            /// * field_has_align: u32   // if `any_aligned_fields`; for every 32 fields; LSB is first field
            /// * field_has_default: u32 // if `any_default_inits`; for every 32 fields; LSB is first field
            /// * fields: { // for each `fields_len`
            ///     name: NullTerminatedString,
            ///     type_body_start: Inst.Index,
            ///     align_body_start: Inst.Index,  // if field has align
            ///     init_body_start: Inst.Index,   // if field has init
            ///   }
            ///
            /// The first `decls_len` many instructions directly following this are all `declaration` instructions.
            /// (Their sub-bodies are placed after the block of `declaration` instructions.)
            pub const StructDecl = struct {
                /// The instruction indices following a `StructDecl` define its fields, declarations, etc, in nested bodies.
                /// The parent body continues at this index.
                next_inst: Inst.Index,

                // These fields should be stored in an array and bitcast to a `std.zig.SrcHash`.
                // This hash contains the source of all fields, and any specified attributes (`extern`, backing type, etc).
                fields_hash_0: u32,
                fields_hash_1: u32,
                fields_hash_2: u32,
                fields_hash_3: u32,

                src_line: u32,

                pub const Small = packed struct(u16) {
                    any_captures: bool,
                    any_fields: bool,
                    any_decls: bool,

                    any_comptime_fields: bool,
                    any_aligned_fields: bool,
                    any_default_inits: bool,

                    has_backing_int: bool,
                    layout: std.builtin.Type.ContainerLayout,
                    name_strategy: NameStrategy,

                    known_non_opv: bool,
                    known_comptime_only: bool,

                    _: u3 = 0,
                };

                pub const FieldFlags = packed struct(u4) {
                    has_align: bool,
                    has_default: bool,
                    has_type: bool,
                    is_comptime: bool,
                };
            };
        };
    };

    pub const Index = enum(u32) {
        root = 0,
        _,
        pub fn get(idx: Index, zir: *const Zir) Inst {
            const repr = zir.instructions.get(@intFromEnum(idx));
            return switch (repr.tag) {
                .int => .{ .int = @bitCast(repr.data) },
                .float64 => .{ .float = @as(f64, @bitCast(repr.data)) },
                .str => .{ .str = .{
                    .index = repr.data[0],
                    .len = repr.data[1],
                } },
                inline .add,
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
                => |tag| @unionInit(Inst, @tagName(tag), .{
                    .lhs = @enumFromInt(repr.data[0]),
                    .rhs = @enumFromInt(repr.data[1]),
                }),
                .coerce => .{ .coerce = .{
                    .operand = @enumFromInt(repr.data[0]),
                    .dest_ty = @enumFromInt(repr.data[1]),
                } },
                .store => .{ .store = .{
                    .ptr = @enumFromInt(repr.data[0]),
                    .operand = @enumFromInt(repr.data[1]),
                } },
                .import => .{ .import = .{
                    .res_ty = @enumFromInt(repr.data[0]),
                    .path = @enumFromInt(repr.data[1]),
                } },
                .block => .{ .block = .{
                    .body = .{ .first_inst = @enumFromInt(@intFromEnum(idx) + 1) },
                    .next_inst = @enumFromInt(repr.data[0]),
                } },
                .@"break" => .{ .@"break" = .{
                    .target = @enumFromInt(repr.data[0]),
                    .operand = @enumFromInt(repr.data[1]),
                } },
                .extended => {
                    const extended: Repr.Extended = @bitCast(repr.data[0]);
                    return switch (extended.tag) {
                        .value_placeholder => unreachable,
                        .struct_decl => {
                            const small: Repr.Extended.StructDecl.Small = @bitCast(extended.small);
                            var extra_idx: ExtraIndex = @enumFromInt(repr.data[1]);
                            const struct_decl = extra_idx.readStructAdvance(zir, Repr.Extended.StructDecl);

                            const captures_len: u32 = if (small.any_captures) extra_idx.readAdvance(zir, u32) else 0;
                            const fields_len: u32 = if (small.any_fields) extra_idx.readAdvance(zir, u32) else 0;
                            const decls_len: u32 = if (small.any_decls) extra_idx.readAdvance(zir, u32) else 0;

                            const captures = extra_idx.sliceAdvance(zir, captures_len, TypeCapture);
                            const capture_names = extra_idx.sliceAdvance(zir, captures_len, NullTerminatedString);

                            const backing_int_body_start: Inst.Index = if (small.has_backing_int) extra_idx.readAdvance(zir, Inst.Index) else undefined;

                            const field_is_comptime_bits: FlagBits = .initAdvance(&extra_idx, small.any_comptime_fields, fields_len);
                            const field_has_align_bits: FlagBits = .initAdvance(&extra_idx, small.any_aligned_fields, fields_len);
                            const field_has_default_bits: FlagBits = .initAdvance(&extra_idx, small.any_default_inits, fields_len);

                            const fields_data_start = extra_idx;
                            extra_idx = undefined; // we can't quickly advnace through the fields, there must be nothing more

                            return .{ .struct_decl = .{
                                .next_inst = struct_decl.next_inst,
                                .fields_hash = @bitCast(@as([4]u32, .{
                                    struct_decl.fields_hash_0,
                                    struct_decl.fields_hash_1,
                                    struct_decl.fields_hash_2,
                                    struct_decl.fields_hash_3,
                                })),
                                .src_line = struct_decl.src_line,
                                .captures = captures,
                                .capture_names = capture_names,
                                .backing_int_body = if (small.has_backing_int) .{ .first_inst = backing_int_body_start } else null,
                                .decls = .{
                                    .first = @enumFromInt(@intFromEnum(idx) + 1),
                                    .len = decls_len,
                                },
                                .fields = .{
                                    .len = fields_len,
                                    .is_comptime_bits = field_is_comptime_bits,
                                    .has_align_bits = field_has_align_bits,
                                    .has_default_bits = field_has_default_bits,
                                    .data_start = fields_data_start,
                                },
                            } };
                        },
                        .decl_ref => .{ .decl_ref = @enumFromInt(repr.data[1]) },
                        .load => .{ .load = @enumFromInt(repr.data[1]) },
                        .negate => .{ .negate = @enumFromInt(repr.data[1]) },
                        .int_big => .{ .int_big = .{
                            .first = repr.data[1],
                            .len = extended.small,
                        } },
                        .float128 => .{ .float = @bitCast(zir.extra[repr.data[1]..][0..4].*) },
                        .validate_const => .{ .validate_const = @enumFromInt(repr.data[1]) },
                        .validate_const_ref => .{ .validate_const_ref = @enumFromInt(repr.data[1]) },
                        .ref => .{ .ref = @enumFromInt(repr.data[1]) },
                        .ensure_result_used => .{ .ensure_result_used = @enumFromInt(repr.data[1]) },
                        .int_type => .{ .int_type = .{
                            .bits = extended.small,
                            .signedness = @enumFromInt(repr.data[1]),
                        } },
                        .zirgen_error => .zirgen_error,
                        .decl_usingnamespace,
                        .decl_comptime,
                        .decl_unnamed_test,
                        => {
                            var extra_idx: ExtraIndex = @enumFromInt(repr.data[1]);
                            const decl = extra_idx.readStructAdvance(zir, Repr.Extended.DeclSimple);
                            return .{ .declaration = .{
                                .src_hash = @bitCast(@as([4]u32, .{
                                    decl.src_hash_0,
                                    decl.src_hash_1,
                                    decl.src_hash_2,
                                    decl.src_hash_3,
                                })),
                                .src_line = decl.src_line,
                                .src_column = decl.src_column,
                                .kind = switch (extended.tag) {
                                    .decl_usingnamespace => .@"usingnamespace",
                                    .decl_comptime => .@"comptime",
                                    .decl_unnamed_test => .unnamed_test,
                                    else => unreachable,
                                },
                                .name = .empty,
                                .is_pub = switch (extended.tag) {
                                    .decl_usingnamespace => switch (extended.small) {
                                        0 => false,
                                        1 => true,
                                        else => unreachable,
                                    },
                                    .decl_comptime, .decl_unnamed_test => false,
                                    else => unreachable,
                                },
                                .is_threadlocal = false,
                                .linkage = .normal,
                                .lib_name = .empty,
                                .value_body = .{ .first_inst = decl.body_start },
                                .type_body = null,
                                .align_body = null,
                                .linksection_body = null,
                                .addrspace_body = null,
                                .fn_info = null,
                            } };
                        },
                        .decl_named_test,
                        .decl_decltest,
                        => {
                            var extra_idx: ExtraIndex = @enumFromInt(repr.data[1]);
                            const decl = extra_idx.readStructAdvance(zir, Repr.Extended.DeclNamedTest);
                            return .{ .declaration = .{
                                .src_hash = @bitCast(@as([4]u32, .{
                                    decl.src_hash_0,
                                    decl.src_hash_1,
                                    decl.src_hash_2,
                                    decl.src_hash_3,
                                })),
                                .src_line = decl.src_line,
                                .src_column = decl.src_column,
                                .kind = switch (extended.tag) {
                                    .decl_named_test => .@"test",
                                    .decl_decltest => .decltest,
                                    else => unreachable,
                                },
                                .name = decl.name,
                                .is_pub = false,
                                .is_threadlocal = false,
                                .linkage = .normal,
                                .lib_name = .empty,
                                .value_body = .{ .first_inst = decl.body_start },
                                .type_body = null,
                                .align_body = null,
                                .linksection_body = null,
                                .addrspace_body = null,
                                .fn_info = null,
                            } };
                        },
                        .decl_const,
                        .decl_var,
                        => {
                            const small: Repr.Extended.DeclConstVar.Small = @bitCast(extended.small);
                            var extra_idx: ExtraIndex = @enumFromInt(repr.data[1]);
                            const decl = extra_idx.readStructAdvance(zir, Repr.Extended.DeclConstVar);
                            const value_body: ?Body = if (small.linkage != .@"extern") .{
                                .first_inst = extra_idx.readAdvance(zir, Inst.Index),
                            } else null;
                            const type_body: ?Body = if (small.has_type) .{
                                .first_inst = extra_idx.readAdvance(zir, Inst.Index),
                            } else null;
                            const align_body: ?Body = if (small.has_align) .{
                                .first_inst = extra_idx.readAdvance(zir, Inst.Index),
                            } else null;
                            const linksection_body: ?Body = if (small.has_linksection) .{
                                .first_inst = extra_idx.readAdvance(zir, Inst.Index),
                            } else null;
                            const addrspace_body: ?Body = if (small.has_addrspace) .{
                                .first_inst = extra_idx.readAdvance(zir, Inst.Index),
                            } else null;
                            const lib_name: NullTerminatedString = if (small.has_lib_name)
                                extra_idx.readAdvance(zir, NullTerminatedString)
                            else
                                .empty;
                            return .{ .declaration = .{
                                .src_hash = @bitCast(@as([4]u32, .{
                                    decl.src_hash_0,
                                    decl.src_hash_1,
                                    decl.src_hash_2,
                                    decl.src_hash_3,
                                })),
                                .src_line = decl.src_line,
                                .src_column = decl.src_column,
                                .kind = switch (extended.tag) {
                                    .decl_const => .@"const",
                                    .decl_var => .@"var",
                                    else => unreachable,
                                },
                                .name = decl.name,
                                .is_pub = small.is_pub,
                                .is_threadlocal = small.is_threadlocal,
                                .linkage = small.linkage,
                                .lib_name = lib_name,
                                .value_body = value_body,
                                .type_body = type_body,
                                .align_body = align_body,
                                .linksection_body = linksection_body,
                                .addrspace_body = addrspace_body,
                                .fn_info = null,
                            } };
                        },
                        .decl_fn => {
                            const small: Repr.Extended.DeclFn.Small = @bitCast(extended.small);
                            var extra_idx: ExtraIndex = @enumFromInt(repr.data[1]);
                            const decl = extra_idx.readStructAdvance(zir, Repr.Extended.DeclFn);
                            const align_body: ?Body = if (small.has_align) .{
                                .first_inst = extra_idx.readAdvance(zir, Inst.Index),
                            } else null;
                            const linksection_body: ?Body = if (small.has_linksection) .{
                                .first_inst = extra_idx.readAdvance(zir, Inst.Index),
                            } else null;
                            const addrspace_body: ?Body = if (small.has_addrspace) .{
                                .first_inst = extra_idx.readAdvance(zir, Inst.Index),
                            } else null;
                            const callconv_body: ?Body = if (small.has_callconv) .{
                                .first_inst = extra_idx.readAdvance(zir, Inst.Index),
                            } else null;
                            const lib_name: NullTerminatedString = if (small.has_lib_name)
                                extra_idx.readAdvance(zir, NullTerminatedString)
                            else
                                .empty;
                            const body: ?Body = if (small.linkage != .@"extern") .{
                                .first_inst = extra_idx.readAdvance(zir, Inst.Index),
                            } else null;

                            // Skip params; they're handled by `Declaration.FnParams.Iterator.next`.
                            const params_data_start = extra_idx;
                            extra_idx.advance(decl.params_len * 3);

                            const param_is_comptime_bits: FlagBits = .initAdvance(&extra_idx, small.any_comptime_params, decl.params_len);
                            const param_is_noalias_bits: FlagBits = .initAdvance(&extra_idx, small.any_noalias_params, decl.params_len);
                            const param_ty_is_generic_bits: FlagBits = .initAdvance(&extra_idx, small.any_generic_param_ty, decl.params_len);

                            return .{ .declaration = .{
                                .src_hash = @bitCast(@as([4]u32, .{
                                    decl.src_hash_0,
                                    decl.src_hash_1,
                                    decl.src_hash_2,
                                    decl.src_hash_3,
                                })),
                                .src_line = decl.src_line,
                                .src_column = decl.src_column,
                                .kind = .@"fn",
                                .name = decl.name,
                                .is_pub = small.is_pub,
                                .is_threadlocal = false,
                                .linkage = small.linkage,
                                .lib_name = lib_name,
                                .value_body = null,
                                .type_body = null,
                                .align_body = align_body,
                                .linksection_body = linksection_body,
                                .addrspace_body = addrspace_body,
                                .fn_info = .{
                                    .proto_hash = @bitCast(@as([4]u32, .{
                                        decl.proto_hash_0,
                                        decl.proto_hash_1,
                                        decl.proto_hash_2,
                                        decl.proto_hash_3,
                                    })),
                                    .params = .{
                                        .len = decl.params_len,
                                        .type_is_generic_bits = param_ty_is_generic_bits,
                                        .is_comptime_bits = param_is_comptime_bits,
                                        .is_noalias_bits = param_is_noalias_bits,
                                        .data_start = params_data_start,
                                    },
                                    .@"callconv" = if (callconv_body) |b|
                                        .{ .body = b }
                                    else if (small.callconv_inline)
                                        .@"inline"
                                    else
                                        .auto,
                                    .ret_ty_body = .{ .first_inst = decl.ret_ty_body_start },
                                    .ret_ty_is_generic = small.ret_ty_is_generic,
                                    .body = body,
                                    .is_var_args = small.is_var_args,
                                    .is_inferred_error = small.is_inferred_error,
                                    .is_noinline = small.is_noinline,
                                },
                            } };
                        },
                        .ret_implicit => .ret_implicit,
                        .ret => .{ .ret = @enumFromInt(repr.data[1]) },
                    };
                },
            };
        }
        pub const Optional = enum(u32) {
            none = std.math.maxInt(u32),
            _,
            pub fn unwrap(opt: Optional) ?Index {
                return switch (opt) {
                    .none => null,
                    _ => @enumFromInt(@intFromEnum(opt)),
                };
            }
        };
        pub fn toOptional(idx: Index) Optional {
            const opt: Optional = @enumFromInt(@intFromEnum(idx));
            assert(opt != .none);
            return opt;
        }
        pub fn toRef(idx: Index) Ref {
            return @enumFromInt(@intFromEnum(idx) + Ref.index_offset);
        }
    };
};

/// A `Ref` refers to some value. Usually, that value is the result of an `Inst.Index`. However,
/// there are a small number of named `Ref`s which mean specific constant values.
pub const Ref = enum(u32) {
    u0_type,
    i0_type,
    u1_type,
    u8_type,
    i8_type,
    u16_type,
    i16_type,
    u29_type,
    u32_type,
    i32_type,
    u64_type,
    i64_type,
    u80_type,
    u128_type,
    i128_type,
    usize_type,
    isize_type,
    c_char_type,
    c_short_type,
    c_ushort_type,
    c_int_type,
    c_uint_type,
    c_long_type,
    c_ulong_type,
    c_longlong_type,
    c_ulonglong_type,
    c_longdouble_type,
    f16_type,
    f32_type,
    f64_type,
    f80_type,
    f128_type,
    anyopaque_type,
    bool_type,
    void_type,
    type_type,
    anyerror_type,
    comptime_int_type,
    comptime_float_type,
    noreturn_type,
    anyframe_type,
    null_type,
    undefined_type,
    enum_literal_type,
    manyptr_u8_type,
    manyptr_const_u8_type,
    manyptr_const_u8_sentinel_0_type,
    single_const_pointer_to_comptime_int_type,
    slice_const_u8_type,
    slice_const_u8_sentinel_0_type,
    vector_16_i8_type,
    vector_32_i8_type,
    vector_16_u8_type,
    vector_32_u8_type,
    vector_8_i16_type,
    vector_16_i16_type,
    vector_8_u16_type,
    vector_16_u16_type,
    vector_4_i32_type,
    vector_8_i32_type,
    vector_4_u32_type,
    vector_8_u32_type,
    vector_2_i64_type,
    vector_4_i64_type,
    vector_2_u64_type,
    vector_4_u64_type,
    vector_4_f16_type,
    vector_8_f16_type,
    vector_2_f32_type,
    vector_4_f32_type,
    vector_8_f32_type,
    vector_2_f64_type,
    vector_4_f64_type,
    optional_noreturn_type,
    anyerror_void_error_union_type,
    adhoc_inferred_error_set_type,
    generic_poison_type,
    empty_tuple_type,
    undef,
    zero,
    zero_usize,
    zero_u8,
    one,
    one_usize,
    one_u8,
    four_u8,
    negative_one,
    void_value,
    unreachable_value,
    null_value,
    bool_true,
    bool_false,
    empty_tuple,

    /// This `Ref` does not correspond to any ZIR instruction or constant value and may instead be
    /// used as a sentinel to indicate null.
    none = std.math.maxInt(u32),

    _,

    /// The offset to apply to an `Inst.Index` to convert it to a `Ref`.
    /// The `- 1` is because of `Ref.none`.
    const index_offset = @typeInfo(Ref).@"enum".fields.len - 1;

    pub fn toIndex(ref: Ref) ?Inst.Index {
        // TODO: I want to do this, but `_` and `else` prongs aren't allowed together.
        //return switch (ref) {
        //    _ => @enumFromInt(@intFromEnum(ref) - index_offset),
        //    else => null,
        //};
        if (@intFromEnum(ref) >= index_offset and ref != .none) {
            return @enumFromInt(@intFromEnum(ref) - index_offset);
        } else {
            return null;
        }
    }
};

pub const BinOp = struct {
    lhs: Ref,
    rhs: Ref,
};
pub const Break = struct {
    target: Inst.Index,
    operand: Ref,
};
pub const Declaration = struct {
    src_hash: std.zig.SrcHash,
    src_line: u32,
    src_column: u32,

    kind: Kind,
    /// Always `.empty` for `kind` of `unnamed_test`, `.@"comptime"`, `.@"usingnamespace"`.
    name: NullTerminatedString,
    /// Always `false` for `kind` of `unnamed_test`, `.@"test"`, `.decltest`, `.@"comptime"`.
    is_pub: bool,
    /// Always `false` for `kind != .@"var"`.
    is_threadlocal: bool,
    /// Always `.normal` for `kind` not equal to `.@"const"`, `.@"var"`, or `.@"fn"`.
    linkage: Linkage,
    /// Always `.empty` for `linkage != .@"extern"`.
    lib_name: NullTerminatedString,

    /// Always populated for `linkage != .@"extern" and kind != .@"fn"`.
    value_body: ?Body,
    /// Always populated for `linkage == .@"extern" and kind != .@"fn"`.
    type_body: ?Body,
    align_body: ?Body,
    linksection_body: ?Body,
    addrspace_body: ?Body,

    /// This is populated instead of `type_body` or `value_body` iff `kind == .@"fn".
    fn_info: ?struct {
        proto_hash: std.zig.SrcHash,
        is_inferred_error: bool,
        is_var_args: bool,
        is_noinline: bool,
        @"callconv": union(enum) {
            auto,
            @"inline",
            body: Body,
        },
        params: FnParams,
        ret_ty_body: Body,
        ret_ty_is_generic: bool,
        /// Populated iff `linkage != .@"extern"`.
        body: ?Body,
    },

    pub const Kind = enum {
        @"const",
        @"var",
        @"fn",

        unnamed_test,
        @"test",
        decltest,

        @"comptime",
        @"usingnamespace",
    };

    pub const Linkage = enum(u2) {
        normal,
        @"extern",
        @"export",
    };

    pub const FnParams = struct {
        len: u32,
        type_is_generic_bits: FlagBits,
        is_comptime_bits: FlagBits,
        is_noalias_bits: FlagBits,
        data_start: ExtraIndex,
        pub fn iterate(params: FnParams, zir: *const Zir) Iterator {
            return .{
                .zir = zir,
                .params_len = params.len,
                .type_is_generic_bits = params.type_is_generic_bits,
                .is_comptime_bits = params.is_comptime_bits,
                .is_noalias_bits = params.is_noalias_bits,
                .extra_idx = params.data_start,
                .next_param_idx = 0,
            };
        }
        pub const Iterator = struct {
            zir: *const Zir,
            params_len: u32,
            type_is_generic_bits: FlagBits,
            is_comptime_bits: FlagBits,
            is_noalias_bits: FlagBits,
            extra_idx: ExtraIndex,
            next_param_idx: u32,
            pub fn next(it: *Iterator) ?Param {
                const param_idx = it.next_param_idx;
                if (param_idx == it.params_len) return null;
                const zir = it.zir;
                var extra_idx = it.extra_idx;
                const result: Param = .{
                    .name = extra_idx.readAdvance(zir, NullTerminatedString),
                    .is_comptime = it.is_comptime_bits.get(zir, param_idx),
                    .is_noalias = it.is_noalias_bits.get(zir, param_idx),
                    .type_is_generic = it.type_is_generic_bits.get(zir, param_idx),
                    .type_body = if (extra_idx.readAdvance(zir, Inst.Index.Optional).unwrap()) |inst| .{
                        .first_inst = inst,
                    } else null,
                    .placeholder_inst = extra_idx.readAdvance(zir, Inst.Index),
                };
                it.extra_idx = extra_idx;
                it.next_param_idx += 1;
                return result;
            }
            pub const Param = struct {
                name: NullTerminatedString,
                is_comptime: bool,
                is_noalias: bool,
                /// This is `true` if the type depends on a prior parameter. It is `false` for `anytype` parameters.
                type_is_generic: bool,
                /// `null` means the parameter is `anytype`.
                type_body: ?Body,
                /// This is a `value_placeholder` instruction which semantic analysis should map to the parameter value.
                placeholder_inst: Inst.Index,
            };
        };
    };
};

pub const Body = struct {
    first_inst: Inst.Index,
    pub fn iterate(body: Body, zir: *const Zir) Iterator {
        return .{ .inst = body.first_inst, .zir = zir };
    }
    pub const Iterator = struct {
        inst: ?Inst.Index,
        zir: *const Zir,
        pub fn next(it: *Iterator) ?Inst.Index {
            const inst = it.inst orelse return null;
            it.inst = it.zir.instAfter(inst);
            return inst;
        }
    };
};

pub const Limbs = struct {
    first: u32,
    len: u16,
    pub fn get(limbs: Limbs, zir: *const Zir) std.math.big.int.Const {
        return .{
            .limbs = zir.limbs[limbs.first..][0..limbs.len],
            .positive = true,
        };
    }
};

pub const Declarations = struct {
    first: Inst.Index,
    len: u32,
    pub fn iterate(decls: Declarations, zir: *const Zir) Iterator {
        return .{
            .zir = zir,
            .next_decl = decls.first,
            .remaining = decls.len,
        };
    }
    pub const Iterator = struct {
        zir: *const Zir,
        next_decl: Inst.Index,
        remaining: u32,
        pub fn next(it: *Iterator) ?Inst.Index {
            if (it.remaining == 0) return null;
            const res = it.next_decl;
            it.next_decl = @enumFromInt(@intFromEnum(res) + 1);
            it.remaining -= 1;
            return res;
        }
    };
};

pub const StructFields = struct {
    len: u32,
    is_comptime_bits: FlagBits,
    has_align_bits: FlagBits,
    has_default_bits: FlagBits,
    data_start: ExtraIndex,
    pub fn iterate(fields: StructFields, zir: *const Zir) Iterator {
        return .{
            .zir = zir,
            .fields_len = fields.len,
            .is_comptime_bits = fields.is_comptime_bits,
            .has_align_bits = fields.has_align_bits,
            .has_default_bits = fields.has_default_bits,
            .extra_idx = fields.data_start,
            .next_field_idx = 0,
        };
    }
    pub const Iterator = struct {
        zir: *const Zir,
        fields_len: u32,
        is_comptime_bits: FlagBits,
        has_align_bits: FlagBits,
        has_default_bits: FlagBits,
        extra_idx: ExtraIndex,
        next_field_idx: u32,
        pub fn next(it: *Iterator) ?Field {
            const field_idx = it.next_field_idx;
            if (field_idx == it.fields_len) return null;
            const zir = it.zir;
            var extra_idx = it.extra_idx;
            const result: Field = .{
                .is_comptime = it.is_comptime_bits.get(zir, field_idx),
                .name = extra_idx.readAdvance(zir, NullTerminatedString),
                .type_body = .{ .first_inst = extra_idx.readAdvance(zir, Inst.Index) },
                .align_body = if (it.has_align_bits.get(zir, field_idx))
                    .{ .first_inst = extra_idx.readAdvance(zir, Inst.Index) }
                else
                    null,
                .init_body = if (it.has_default_bits.get(zir, field_idx))
                    .{ .first_inst = extra_idx.readAdvance(zir, Inst.Index) }
                else
                    null,
            };
            it.extra_idx = extra_idx;
            it.next_field_idx += 1;
            return result;
        }
        pub const Field = struct {
            is_comptime: bool,
            name: NullTerminatedString,
            type_body: Body,
            align_body: ?Body,
            init_body: ?Body,
        };
    };
};

pub const NameStrategy = enum(u2) {
    /// This type is declared in the value body of a declaration; use that declaration's name.
    /// e.g. `const Foo = struct {...};`.
    parent,
    /// Use the name of the currently executing comptime function call, with the current parameters.
    /// e.g. `ArrayList(i32)`.
    func,
    /// Create an anonymous name for this declaration.
    /// Like this: "ParentDeclName_struct_69"
    anon,
    /// Use the name specified in the next `dbg_var_{val,ptr}` instruction.
    dbg_var,
};

pub const TypeCapture = packed struct(u32) {
    _: u32 = 0,
    // TODO
};

pub const NullTerminatedString = enum(u32) {
    /// To avoid an extra branch in `get`, `ZirGen` guarantees a zero byte at the start of `Zir.string_bytes`.
    empty = 0,
    _,
    pub fn get(nts: NullTerminatedString, zir: *const Zir) [:0]const u8 {
        const overlong = zir.string_bytes[@intFromEnum(nts)..];
        const len = std.mem.indexOfScalar(u8, overlong, 0).?;
        return overlong[0..len :0];
    }
};

pub const FlagBits = struct {
    /// `null` means all bits are `false`.
    extra_idx: ?ExtraIndex,
    entries_len: if (std.debug.runtime_safety) u32 else void,
    pub fn get(bits: FlagBits, zir: *const Zir, entry_idx: u32) bool {
        if (std.debug.runtime_safety) {
            assert(entry_idx < bits.entries_len);
        }

        const extra_idx = bits.extra_idx orelse return false;

        const elem_offset = entry_idx / 32;
        const bit_offset = entry_idx % 32;

        const elem = extra_idx.readOffset(zir, elem_offset, u32);
        const bit: u1 = @truncate(elem >> @intCast(bit_offset));
        return @bitCast(bit);
    }
    /// Initialize a `FlagBits` from an `ExtraIndex`, and advance that `ExtraIndex` as needed.
    fn initAdvance(extra_idx: *ExtraIndex, has_bits: bool, entries_len: u32) FlagBits {
        if (!has_bits) return .{
            .extra_idx = null,
            .entries_len = if (std.debug.runtime_safety) entries_len,
        };

        const bits_start = extra_idx.*;
        extra_idx.advance(std.math.divCeil(u32, entries_len, 32) catch unreachable);
        return .{
            .extra_idx = bits_start,
            .entries_len = if (std.debug.runtime_safety) entries_len,
        };
    }
};

pub const ExtraIndex = enum(u32) {
    _,

    fn readStructAdvance(idx: *ExtraIndex, zir: *const Zir, comptime T: type) T {
        var result: T = undefined;
        inline for (@typeInfo(T).@"struct".fields) |field| {
            @field(result, field.name) = idx.readAdvance(zir, field.type);
        }
        return result;
    }
    fn readAdvance(idx: *ExtraIndex, zir: *const Zir, comptime T: type) T {
        const result = idx.readOffset(zir, 0, T);
        idx.* = @enumFromInt(@intFromEnum(idx.*) + 1);
        return result;
    }
    fn sliceAdvance(idx: *ExtraIndex, zir: *const Zir, len: u32, comptime T: type) []const T {
        comptime assert(@sizeOf(T) == 4);
        comptime assert(std.meta.hasUniqueRepresentation(T));
        const slice = zir.extra[@intFromEnum(idx.*)..][0..len];
        idx.* = @enumFromInt(@intFromEnum(idx.*) + len);
        return @ptrCast(slice);
    }
    fn advance(idx: *ExtraIndex, off: u32) void {
        idx.* = @enumFromInt(@intFromEnum(idx.*) + off);
    }

    fn readOffset(idx: ExtraIndex, zir: *const Zir, offset: u32, comptime T: type) T {
        const raw = zir.extra[@intFromEnum(idx) + offset];
        return switch (T) {
            u32 => raw,
            NullTerminatedString, Inst.Index, Inst.Index.Optional => @enumFromInt(raw),
            else => comptime unreachable,
        };
    }
};

/// Given an `Inst.Index`, returns the `Inst.Index` of the instruction immediately following it, or
/// `null` if `idx` is the last instruction in its body.
fn instAfter(zir: *const Zir, idx: Inst.Index) ?Inst.Index {
    return switch (idx.get(zir)) {
        // Instructions which do not appear in bodies
        .declaration,
        => unreachable,

        // Instructions which are always `noreturn`
        .@"break",
        .ret,
        .ret_implicit,
        .zirgen_error,
        => null,

        // Trivial instructions, i.e. those followed by the next index
        .int,
        .int_big,
        .float,
        .str,

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

        .coerce,
        .store,
        .import,

        .decl_ref,
        .load,
        .negate,
        .validate_const,
        .validate_const_ref,
        .ref,
        .ensure_result_used,
        .int_type,
        => @enumFromInt(@intFromEnum(idx) + 1),

        .block => |b| b.next_inst,
        .struct_decl => |sd| sd.next_inst,
    };
}

const std = @import("std");
const Ast = std.zig.Ast;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Zir = @This();
