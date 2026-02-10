import re
from typing import List, Tuple, Dict
from MultiDeeploy.LayerProfile import LayerProfile
from MultiDeeploy.utils import normalize_layer_name, parse_closure_body_sections, split_closure_call_body, split_tiling_call_body, split_typedef_struct, split_function_call, split_closure_L3_call_body, split_compound_literal, split_struct_and_function

class GlobalsDefinition():

    GLOBALS_END_PATTERN = "extern struct pi_device cluster_dev;"

    def __init__(self):
        self.content = []

    def getGlobals(self, content: str) -> None:
        """
        Extracts the global variable declarations from a Network.c file content.
        """
        lines = content.splitlines(keepends=True)

        split_idx = None
        for i, line in enumerate(lines):
            if self.GLOBALS_END_PATTERN in line:
                split_idx = i
                break

        self.content = lines[:split_idx + 1]

class LayerFunctionDefinition:
    """
    Each attribute is:
      [
        [struct_prefix, struct_args, struct_suffix],
        [call_prefix,   call_body,   call_suffix],
      ]
    """

    def __init__(self):
        self.tiling_closure = None
        self.cluster_fork = None
        self.closure = None
        self.closure_L3 = None

        self.tiling_closure_body = []
        self.closure_L3_body = []
        self.cluster_fork_body = []
        self.closure_body = []

    @staticmethod
    def _split_block(block: str) -> list[list[str]]:
        """
        Returns:
        [
            [struct_prefix, struct_args, struct_suffix],
            [call_prefix, call_body, call_suffix],
        ]
        """
        struct_part, call_part = split_struct_and_function(block)
        struct_prefix, struct_args, struct_suffix = split_typedef_struct(struct_part)
        call_prefix, call_body, call_suffix = split_function_call(call_part)

        return [
            [struct_prefix, struct_args, struct_suffix],
            [call_prefix, call_body, call_suffix],
        ]

    def parse_blocks(self, tiling: str, cluster: str, closure: str, closure_L3: str) -> None:
        self.tiling_closure = self._split_block(tiling)
        self.cluster_fork   = self._split_block(cluster)
        self.closure        = self._split_block(closure)
        self.closure_L3     = self._split_block(closure_L3)

    def parse_closure_L3(self):
        parts = split_closure_L3_call_body(self.closure_L3[1][1])

        self.closure_L3_body.append(parts["args_cast"])
        prefix, args, suffix = split_compound_literal(
            parts["closure_function_call"]
        )
        self.closure_L3_body.append([prefix, args, suffix])
        self.closure_L3_body.append(parts["closure_call"])

    def parse_cluster_fork(self):
        parts = split_closure_L3_call_body(self.cluster_fork[1][1])

        self.cluster_fork_body.append(parts["args_cast"])
        prefix, args, suffix = split_compound_literal(
            parts["closure_function_call"]
        )
        self.cluster_fork_body.append([prefix, args, suffix])
        self.cluster_fork_body.append(parts["closure_call"])

    def parse_closure(self):
        parts = split_closure_call_body(self.closure[1][1])

        self.closure_body = [
            parts["arg_cast"],
            parse_closure_body_sections(parts["body"]),
        ]

    def parse_tiling_closure(self):
        parts = split_tiling_call_body(self.tiling_closure[1][1])

        self.tiling_closure_body = [
            parts["arg_cast"],
            parts["function_body"],
        ]

class FunctionsDefinition:

    def __init__(self, content: str):
        self.content = content
        self.layers_functions_definition = []

    def parseContent(self) -> None:
        STRUCT_START  = "// LAYER FUNCTION STRUCT DEFINITION, START"
        STRUCT_END    = "// LAYER FUNCTION STRUCT DEFINITION, END"
        CLOSURE_START = "// LAYER FUNCTION CLOSURE, START"
        CLOSURE_END   = "// LAYER FUNCTION CLOSURE, END"

        blocks = []
        cur_struct = []
        cur_call = []
        mode = None

        for line in self.content.splitlines(keepends=True):

            if STRUCT_START in line:
                mode = "struct"
                cur_struct = []
                continue

            if STRUCT_END in line:
                mode = None
                continue

            if CLOSURE_START in line:
                mode = "call"
                cur_call = []
                continue

            if CLOSURE_END in line:
                mode = None
                blocks.append(("".join(cur_struct), "".join(cur_call)))
                cur_struct = []
                cur_call = []
                continue

            if mode == "struct":
                cur_struct.append(line)
            elif mode == "call":
                cur_call.append(line)

        for i in range(0, len(blocks), 4):
            layer = LayerFunctionDefinition()
            layer.parse_blocks(
                blocks[i + 0][0] + blocks[i + 0][1],
                blocks[i + 1][0] + blocks[i + 1][1],
                blocks[i + 2][0] + blocks[i + 2][1],
                blocks[i + 3][0] + blocks[i + 3][1],
            )
            layer.parse_closure_L3()
            layer.parse_cluster_fork()
            layer.parse_closure()
            layer.parse_tiling_closure()
            self.layers_functions_definition.append(layer)


class FunctionCall():

    RE_LAYER_FROM_CALL = re.compile(r"\b([A-Za-z0-9_]+)_([A-Za-z0-9_]+)_closure(?:_[A-Za-z0-9_]+)?\s*\(")
    RE_ASSIGN_END = re.compile(r";\s*$")
    RE_COMPOUND_START = re.compile(r"\)\s*\{")
    RE_COMPOUND_END_INLINE = re.compile(r"(.*?)(\};)(.*)", re.DOTALL)
    RE_ARGS_STRUCT_START = re.compile(r"^\s*[A-Za-z_]\w*(?:\s+\*+)?\s+[A-Za-z_]\w*\s*=")  # Allows pointers

    def __init__(self, content: str, model_name: str):
        self.content = content
        self.model_name = model_name
        self.layer_name: str = ""

        self.layer_buffer: str = ""
        self.prefix_args_cast: str = ""
        self.args_cast: str = ""
        self.layer_function_call: str = ""

    def parseContent(self) -> None:
        lines = self.content.splitlines(keepends=True)
        i = 0
        n = len(lines)

        self.layer_buffer = ""
        self.prefix_args_cast = ""
        self.args_cast = ""
        self.layer_function_call = ""

        # 1) Optional layer buffer (assignment BUT NOT args struct)
        # Skip comments and blank lines first
        while i < n and (lines[i].strip() == "" or lines[i].lstrip().startswith("//")):
            i += 1

        if (
            i < n
            and "=" in lines[i]
            and not self.RE_ARGS_STRUCT_START.match(lines[i])
            and not lines[i].lstrip().startswith("//")
        ):
            buf = []
            while i < n:
                buf.append(lines[i])
                if ";" in lines[i]:
                    i += 1
                    break
                i += 1
            self.layer_buffer = "".join(buf)

        # 2) Skip blank lines and comments
        while i < n and (lines[i].strip() == "" or lines[i].lstrip().startswith("//")):
            i += 1

        # 3) Args struct prefix (up to and including '{')
        # Collect lines until we find the opening brace '{'
        prefix = []
        prefix_brace_content = ""  # Content after '{' on the same line
        
        while i < n:
            line = lines[i]
            if "{" in line:
                # Split on the opening brace
                before_brace, _, after_brace = line.partition("{")
                prefix.append(before_brace + "{")
                prefix_brace_content = after_brace
                i += 1
                break
            prefix.append(line)
            i += 1
        
        self.prefix_args_cast = "".join(prefix)

        # 4) Args struct body (until '};')
        # Start with any content after '{' on the same line, then collect more lines
        args = []
        if prefix_brace_content:
            args.append(prefix_brace_content)
        
        closing = ""
        
        while i < n:
            line = lines[i]
            if "};" in line:
                before, _, after = line.partition("};")
                if before.strip():
                    args.append(before)
                closing = "};" + after
                i += 1
                break
            args.append(line)
            i += 1

        self.args_cast = "".join(args)

        # 5) Closure call - collect from closing brace until end
        # This includes any comments, blank lines, and the actual function call
        call_lines = []
        call_lines.append(closing)
        call_lines.extend(lines[i:])
        self.layer_function_call = "".join(call_lines)

        # 6) Extract layer name from the function call
        # Try to find the closure function call pattern
        m = self.RE_LAYER_FROM_CALL.search(self.layer_function_call)
        if not m:
            # Fallback: try to extract from prefix_args_cast or args_cast
            m = self.RE_LAYER_FROM_CALL.search(self.prefix_args_cast + self.args_cast)
            if not m:
                raise RuntimeError(
                    f"Could not extract layer name from function call:\n{self.layer_function_call}\n\nOr from prefix:\n{self.prefix_args_cast}"
                )

        if self.model_name == "testRQConv":
            self.layer_name = m.group(1) + "_" + m.group(2)
        else:
            self.layer_name = normalize_layer_name(
                m.group(1) + "_" + m.group(2)
            )



class RunNetworkFunctionDefinition():

    RUN_NETWORK_START_PATTERN = "void RunNetwork("
    INIT_NETWORK_START_PATTERN = "void InitNetwork("
    RE_ASSIGNMENT = re.compile(r"^\s*[\w_]+\s*=\s*.+")  # Assignment (may span multiple lines)
    RE_CLOSURE_CALL = re.compile(r"^\s*[\w_]+\s*\([^;]*\);\s*$")
    RE_ARGS_STRUCT_START = re.compile(r"^\s*[A-Za-z_]\w*(?:\s+\*+)?\s+[A-Za-z_]\w*\s*=")  # Allows pointers

    def __init__(self, model_name: str):
        self.content = []
        self.model_name = model_name
        
        self.variables: List[str] = []
        self.functions_calls = []

    def getRunNetworkFunction(self, content: str) -> None:
        """
        Extracts the RunNetwork function content from a Network.c file content.
        """
        lines = content.splitlines(keepends=True)

        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            if self.RUN_NETWORK_START_PATTERN in line:
                start_idx = i + 1
                continue
            if start_idx is not None and self.INIT_NETWORK_START_PATTERN in line:
                end_idx = i - 1
                break

        self.content = lines[start_idx:end_idx]

    def getVariablesDefinitions(self) -> None:
        """
        Extracts variable definitions from the RunNetwork function definition.
        """
        stmt = []
        cursor = 0

        for i, line in enumerate(self.content):
            stmt.append(line)

            if ";" not in line:
                continue
            full_stmt = "".join(stmt)
            if re.match(
                r"^\s*(?:const\s+)?(?:struct\s+)?[A-Za-z_]\w*(?:\s*\*+|\s+)\s*[A-Za-z_]\w*",
                full_stmt,
            ):
                self.variables.append(full_stmt)
                stmt = []
                cursor = i + 1
                continue

            # First non-variable statement, we stop
            break

        self._vars_end_idx = cursor


    def getFunctionCalls(self) -> None:
        """
        Extracts execution blocks (buffer + args cast + closure call)
        into FunctionCall objects.

        Supports:
        - assignment + call
        - args struct + call
        - bare closure calls (no preamble)
        - multi-line calls

        Note: We keep the original line-based parsing for all blocks,
        and only post-fix the very last block (which can have a multiline call).
        """

        start_idx = self._vars_end_idx
        lines = self.content[start_idx:]

        current_block: List[str] = []
        inside_block = False
        saw_call_start = False

        RE_CALL_START = re.compile(r"^\s*[\w_]+\s*\(")
        RE_CALL_END   = re.compile(r"\);\s*$")

        # --- Original logic: KEEP AS-IS for all "normal" blocks ---
        for line in lines:
            # 1) Start block on ASSIGNMENT or ARGS STRUCT
            if not inside_block and (
                self.RE_ASSIGNMENT.match(line)
                or self.RE_ARGS_STRUCT_START.match(line)
            ):
                inside_block = True
                saw_call_start = False
                current_block = [line]
                continue

            # 2) Start block on BARE CALL (no preamble)
            if not inside_block and RE_CALL_START.match(line):
                inside_block = True
                saw_call_start = True
                current_block = [line]

                # single-line bare call
                if RE_CALL_END.search(line):
                    self.functions_calls.append(
                        FunctionCall("".join(current_block), self.model_name)
                    )
                    current_block = []
                    inside_block = False
                    saw_call_start = False
                continue

            # 3) Ignore lines until a block starts
            if not inside_block:
                continue

            # 4) Inside block → collect lines
            current_block.append(line)

            if RE_CALL_START.match(line):
                saw_call_start = True

            # 5) End block on call end
            if saw_call_start and RE_CALL_END.search(line):
                self.functions_calls.append(
                    FunctionCall("".join(current_block), self.model_name)
                )
                current_block = []
                inside_block = False
                saw_call_start = False

        # Keep the original safety net behavior for truly broken parses,
        # BUT we will still attempt a "last block" fix below even if current_block is empty.
        if current_block:
            raise RuntimeError(
                f"Unterminated function call block:\n{''.join(current_block)}"
            )

        # --- Post-pass: ONLY fix the last block, parsed differently ---
        # Invariant you described: the final call always has the same shape and may be multiline.
        #
        # We reconstruct it from:
        #   last args struct declaration  -->  last line that ends a call (');')
        #
        # Then we replace the last FunctionCall if it was captured as a bare call, or append otherwise.

        if not lines:
            return

        # Heuristic patterns for the last args struct and last call end
        # (use simple, reliable anchors; avoid relying on "__" conventions)
        RE_LAST_ARGS_DECL = re.compile(r"^\s*[A-Za-z_]\w*(?:_[A-Za-z0-9_]+)*_args_t\b")
        RE_LAST_CALL_END  = re.compile(r"\);\s*$")

        last_args_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if RE_LAST_ARGS_DECL.search(lines[i]) or "_args_t" in lines[i]:
                last_args_idx = i
                break

        last_call_end_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if RE_LAST_CALL_END.search(lines[i]):
                last_call_end_idx = i
                break

        # If we can't confidently find both anchors, do nothing (leave the parsed result as-is)
        if last_args_idx is None or last_call_end_idx is None or last_call_end_idx < last_args_idx:
            return

        last_block = "".join(lines[last_args_idx:last_call_end_idx + 1])

        # If the last parsed block is just a bare call, replace it; otherwise append
        if self.functions_calls:
            last_parsed = self.functions_calls[-1].content
            last_parsed_is_bare_call = (
                "_args_t" not in last_parsed
                and RE_CALL_START.search(last_parsed) is not None
            )

            if last_parsed_is_bare_call:
                self.functions_calls[-1] = FunctionCall(last_block, self.model_name)
            else:
                # Avoid duplicating if we already captured it correctly
                if last_block.strip() != last_parsed.strip():
                    self.functions_calls.append(FunctionCall(last_block, self.model_name))
        else:
            self.functions_calls.append(FunctionCall(last_block, self.model_name))
    
    def deduplicateFunctionCallsByLayer(self) -> None:
        """
        Removes duplicate FunctionCall objects based on layer_name.
        Keeps the first occurrence and preserves order.

        Assumes FunctionCall.parseContent() has already been called.
        """

        seen_layers = set()
        unique_calls = []

        for fc in self.functions_calls:
            if fc.layer_name in seen_layers:
                continue
            seen_layers.add(fc.layer_name)
            unique_calls.append(fc)

        self.functions_calls = unique_calls

class InitNetworkFunctionDefinition():

    INIT_NETWORK_START_PATTERN = "void InitNetwork("

    def __init__(self):
        self.content = []

    def getInitNetworkFunction(self, content: str) -> None:
        """
        Extracts the InitNetwork function content from a Network.c file content.
        """
        lines = content.splitlines(keepends=True)

        split_idx = None
        for i, line in enumerate(lines):
            if self.INIT_NETWORK_START_PATTERN in line:
                split_idx = i
                break

        self.content = lines[split_idx + 1:-1]

class SourceFile():

    def __init__(self, path: str, content: str, model_name: str, layers: List[LayerProfile]):
        self.path = path
        self.content = content
        self.model_name = model_name
        self.layers = layers

        self.globals = GlobalsDefinition()
        self.functions = FunctionsDefinition(self.content)
        self.run_network = RunNetworkFunctionDefinition(self.model_name)
        self.init_network = InitNetworkFunctionDefinition()

    def getGlobals(self):
        self.globals.getGlobals(self.content)

    def getFunctions(self):
        self.functions.parseContent()

    def getRunNetworkFunction(self):
        self.run_network.getRunNetworkFunction(self.content)
        self.run_network.getVariablesDefinitions()
        self.run_network.getFunctionCalls()
        for func_call in self.run_network.functions_calls:
            # print(f"[SourceFile] Function call: {func_call.content}")
            func_call.parseContent()
        self.run_network.deduplicateFunctionCallsByLayer()

    def getInitNetworkFunction(self):
        self.init_network.getInitNetworkFunction(self.content)