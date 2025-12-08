
class Command(object):
    def execute(self):
        raise NotImplementedError()
    
    def undo(self):
        raise NotImplementedError()

    def description(self):
        return "Command"

class History(object):
    def __init__(self, limit=100):
        self.undo_stack = []
        self.redo_stack = []
        self.limit = limit
    
    def push(self, command):
        self.undo_stack.append(command)
        if len(self.undo_stack) > self.limit:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        
    def undo(self):
        if not self.undo_stack:
            return None
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        return command
        
    def redo(self):
        if not self.redo_stack:
            return None
        command = self.redo_stack.pop()
        command.execute()
        self.undo_stack.append(command)
        return command
    
    def can_undo(self):
        return len(self.undo_stack) > 0
        
    def can_redo(self):
        return len(self.redo_stack) > 0

class MoveColumnCommand(Command):
    def __init__(self, rompar, idx, dx, relative=True, indices=None, handle_type=None):
        self.rompar = rompar
        self.idx = idx
        self.dx = dx
        self.relative = relative
        self.final_idx = None
        self.indices = indices
        self.final_indices = None
        self.handle_type = handle_type
        
    def execute(self):
        target_indices = self.indices if self.indices else [self.idx]
        res_map = self.rompar._move_bit_column_internal(self.idx, self.dx, self.relative, indices=target_indices, handle_type=self.handle_type)
        if res_map:
             self.final_indices = [res_map[i] for i in target_indices if i in res_map]
             if self.idx in res_map: self.final_idx = res_map[self.idx]
        return res_map is not None

    def undo(self):
        if self.final_indices:
             # Reverse move
             self.rompar._move_bit_column_internal(self.final_indices[0], -self.dx, self.relative, indices=self.final_indices, handle_type=self.handle_type)

    def description(self):
        return f"Move Col {self.idx}" + ("+" if self.indices and len(self.indices)>1 else "")

class MoveRowCommand(Command):
    def __init__(self, rompar, idx, dy, relative=True, indices=None, handle_type=None):
        self.rompar = rompar
        self.idx = idx
        self.dy = dy
        self.relative = relative
        self.final_idx = None
        self.indices = indices
        self.final_indices = None
        self.handle_type = handle_type

    def execute(self):
        target_indices = self.indices if self.indices else [self.idx]
        res_map = self.rompar._move_bit_row_internal(self.idx, self.dy, self.relative, indices=target_indices, handle_type=self.handle_type)
        if res_map:
             self.final_indices = [res_map[i] for i in target_indices if i in res_map]
             if self.idx in res_map: self.final_idx = res_map[self.idx]
        return res_map is not None

    def undo(self):
        if self.final_indices:
             self.rompar._move_bit_row_internal(self.final_indices[0], -self.dy, self.relative, indices=self.final_indices, handle_type=self.handle_type)

    def description(self):
         return f"Move Row {self.idx}" + ("+" if self.indices and len(self.indices)>1 else "")

    def description(self):
        return f"Move Row {self.idx}"

class ToggleBitCommand(Command):
    def __init__(self, rompar, bit_xy, old_val):
        self.rompar = rompar
        self.bit_xy = bit_xy
        self.old_val = old_val
        self.new_val = not old_val

    def execute(self):
        self.rompar.set_data(self.bit_xy, self.new_val)

    def undo(self):
        self.rompar.set_data(self.bit_xy, self.old_val)
        
    def description(self):
        return f"Toggle Bit {self.bit_xy}"

class AddColumnCommand(Command):
    def __init__(self, rompar, img_x):
        self.rompar = rompar
        self.img_x = img_x
        self.added_idx = None

    def execute(self):
        self.added_idx = self.rompar._add_bit_column_internal(self.img_x)
        return self.added_idx is not None

    def undo(self):
        if self.added_idx is not None:
            self.rompar._del_bit_column_internal(self.added_idx)

    def description(self):
        return "Add Col"

class DeleteColumnCommand(Command):
    def __init__(self, rompar, idx):
        self.rompar = rompar
        self.idx = idx
        self.saved_line = None
        self.saved_data = None

    def execute(self):
        result = self.rompar._del_bit_column_internal(self.idx)
        if result:
            self.saved_line, self.saved_data = result
        return result is not None

    def undo(self):
        if self.saved_line and self.saved_data is not None:
             self.rompar._restore_bit_column_internal(self.idx, self.saved_line, self.saved_data)

    def description(self):
        return f"Del Col {self.idx}"

class AddRowCommand(Command):
    def __init__(self, rompar, img_y):
        self.rompar = rompar
        self.img_y = img_y
        self.added_idx = None

    def execute(self):
        self.added_idx = self.rompar._add_bit_row_internal(self.img_y)
        return self.added_idx is not None

    def undo(self):
        if self.added_idx is not None:
            self.rompar._del_bit_row_internal(self.added_idx)

    def description(self):
        return "Add Row"

class DeleteRowCommand(Command):
    def __init__(self, rompar, idx):
        self.rompar = rompar
        self.idx = idx
        self.saved_line = None
        self.saved_data = None

    def execute(self):
        result = self.rompar._del_bit_row_internal(self.idx)
        if result:
            self.saved_line, self.saved_data = result
        return result is not None

    def undo(self):
        if self.saved_line and self.saved_data is not None:
             self.rompar._restore_bit_row_internal(self.idx, self.saved_line, self.saved_data)

    def description(self):
        return f"Del Row {self.idx}"
