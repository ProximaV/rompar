#! /usr/bin/env python3
import traceback
import pathlib
import json

from .. import Rompar, Config, ImgXY
from rompar.util import json_load_exit_bad, exit_message

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

from .about import RomparAboutDialog
from .findhexdialog import FindHexDialog

# Parse the ui file once.
from rompar.history import MoveColumnCommand, MoveRowCommand

import sys, os.path
from PyQt5 import uic
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(thisdir) # Needed to load ui
RomparUi, RomparUiBase = uic.loadUiType(os.path.join(thisdir, 'main.ui'))
del sys.path[-1] # Remove the now unnecessary path entry

MODE_EDIT_GRID = 0
MODE_EDIT_DATA = 1

class RomparUiQt(QtWidgets.QMainWindow):
    def __init__(self, config, *, img_fn=None, grid_fn=None,
                 group_cols=0, group_rows=0, txt=None, annotate=None):
        super(RomparUiQt, self).__init__()
        self.ui = RomparUi()
        self.ui.setupUi(self)
        
        # Add secondary shortcuts for Arrow Keys
        self.ui.actionMoveColumnLeft.setShortcuts([QtGui.QKeySequence("Ctrl+Left"), QtGui.QKeySequence("Left")])
        self.ui.actionMoveColumnRight.setShortcuts([QtGui.QKeySequence("Ctrl+Right"), QtGui.QKeySequence("Right")])
        self.ui.actionMoveRowUp.setShortcuts([QtGui.QKeySequence("Ctrl+Up"), QtGui.QKeySequence("Up")])
        self.ui.actionMoveRowDown.setShortcuts([QtGui.QKeySequence("Ctrl+Down"), QtGui.QKeySequence("Down")])
        
        self.drag_start_pos = None
        self.drag_start_index = None
        self.drag_axis = None
        self.did_drag_select_add = False
        self.drag_start_selection = None

        self.config = config
        self.grid_fn = pathlib.Path(grid_fn).expanduser().absolute() \
                       if grid_fn else None
        grid_json = None
        grid_dir_path = None

        if self.grid_fn:
            print("loading", self.grid_fn)
            grid_json = json_load_exit_bad(str(self.grid_fn), "--load")
            grid_dir_path = self.grid_fn.parent

        self.romp = Rompar(config,
                           img_fn=img_fn, grid_json=grid_json,
                           group_cols=group_cols, group_rows=group_rows,
                           grid_dir_path=grid_dir_path,
                           annotate=annotate)
        self.saven = 0

        # QT Designer doesn't support adding buttons to the taskbar.
        # This moves a button at the botton of the window to the taskbar,
        self.statusBar().addPermanentWidget(self.ui.buttonToggleMode)

        # Make the edit mode exclusive.
        self.mode_selection = QtWidgets.QActionGroup(self)
        self.mode_selection.addAction(self.ui.actionGridEditMode)
        self.mode_selection.addAction(self.ui.actionDataEditMode)
        self.mode_selection.exclusive = True
        
        # Drag State
        self.dragging_handle = False
        self.last_mouse_pos = None
        
        # Box Selection State
        self.selecting_box = False
        self.drag_start_pos_box = None
        
        # Install Event Filter for Dragging
        self.ui.graphicsView.viewport().installEventFilter(self)

        # Make the Image BG selection exclusive.
        self.baseimage_selection = QtWidgets.QActionGroup(self)
        self.baseimage_selection.addAction(self.ui.actionImgBGBlank)
        self.baseimage_selection.addAction(self.ui.actionImgBGOriginal)
        self.baseimage_selection.addAction(self.ui.actionImgBGTarget)
        self.baseimage_selection.exclusive = True

        # Note: This depends on the img_display selection order in
        # rommpar.render_image.
        if self.config.img_display_blank_image:
            self.ui.actionImgBGBlank.setChecked(True)
        elif self.config.img_display_original:
            self.ui.actionImgBGOriginal.setChecked(True)
        else:
            self.ui.actionImgBGTarget.setChecked(True)

        # Set initial state for the various check boxes.
        self.ui.actionShowGrid.setChecked(self.config.img_display_grid)
        self.ui.actionShowDataBinary.setChecked(self.config.img_display_binary)
        self.ui.actionShowPeephole.setChecked(self.config.img_display_peephole)
        self.ui.actionShowData.setChecked(self.config.img_display_data)
        self.ui.actionDataInverted.setChecked(self.config.inverted)
        self.ui.actionDataLSBitMode.setChecked(self.config.LSB_Mode)

        # Create buffers to show Rompar image in UI.
        self.pixmapitem = QtWidgets.QGraphicsPixmapItem()
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.addItem(self.pixmapitem)
        
        # Selection Box Item (Hidden by default)
        self.selection_box_item = QtWidgets.QGraphicsRectItem()
        self.selection_box_item.setPen(QtGui.QPen(QtCore.Qt.cyan))
        self.selection_box_item.setZValue(100) # Ensure it is on top
        self.selection_box_item.hide()
        self.selection_box_item.setZValue(100) # Ensure it is on top
        self.selection_box_item.hide()
        self.scene.addItem(self.selection_box_item)
        
        # Grid Drag Overlays
        self.drag_lines = [] # List of (QGraphicsLineItem, original_line_obj)

        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.setAlignment(QtCore.Qt.AlignTop|QtCore.Qt.AlignLeft)

        # Must be loaded before initial draw
        if txt:
            self.romp.load_txt_data(open(txt, "r"))

        # Do initial draw
        self.img = self.romp.render_image(rgb=True)
        self.qImg = QtGui.QImage(self.img.data,
                                 self.romp.img_width, self.romp.img_height,
                                 self.romp.img_channels * self.romp.img_width,
                                 QtGui.QImage.Format_RGB888)
        self.pixmapitem.setPixmap(QtGui.QPixmap(self.qImg))

        if self.grid_fn:
             self.set_edit_mode(MODE_EDIT_DATA)
        else:
             self.set_edit_mode(MODE_EDIT_GRID)
             self.ui.actionSave.setEnabled(False)
             self.ui.actionBackupSave.setEnabled(False)

    def display_image(self, viewport=None, fast=False):
        self.romp.render_image(img_display=self.img, rgb=True, viewport=viewport, fast=fast)
        self.pixmapitem.setPixmap(QtGui.QPixmap(self.qImg))

    def showTempStatus(self, *msg):
        full_msg = " ".join((str(part) for part in msg))
        print("Status:", repr(full_msg))
        self.statusBar().showMessage(full_msg, 4000)

    def next_save(self):
        '''Look for next unused save slot by checking grid files'''
        while True:
            fn = self.grid_fn.with_suffix(".s%d.json" % self.saven)
            if not fn.is_file():
                return fn
            self.saven += 1

    def save_grid(self, backup=False):
        backup_fn = None
        if backup:
            backup_fn = self.next_save()
            self.grid_fn.rename(backup_fn)

        try:
            with self.grid_fn.open('w') as f:
                json.dump(self.romp.dump_grid_configuration(self.grid_fn.parent),
                          f, indent=4, sort_keys=True)

            # Enable the save options once a save succeeded
            self.ui.actionSave.setEnabled(True)
            self.ui.actionBackupSave.setEnabled(True)

            self.showTempStatus('Saved Grid %s (%s)' % \
                                (str(self.grid_fn),
                                 ("Backed Up: %s" % str(backup_fn)) if backup
                                 else "No Back Up"))
        except Exception as e:
            if backup_fn:
                backup_fn.rename(self.grid_fn) # Restore backup
            QtWidgets.QMessageBox.warning(self, "Error Saving '%s'"%(self.grid_fn),
                                          traceback.format_exc())
            return False
        return True

    def save_data_as_text(self, filepath):
        try:
            with filepath.open('w') as f:
                self.romp.write_data_as_txt(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self,"Error Saving '%s'"%(filepath),
                                          traceback.format_exc())
            return False
        return True

    def shift_xy(self, dx, dy):
        self.romp.shift_xy(dx, dy)
        self.romp.redraw_grid()
        self.display_image()

    def set_edit_mode(self, mode):
        if mode == MODE_EDIT_GRID:
            print("Changing edit mode to GRID")
            self.mode = MODE_EDIT_GRID
            self.ui.buttonToggleMode.setChecked(True)
            self.romp.set_grid_mode(True)
        else:
            print("Changing edit mode to DATA")
            self.mode = MODE_EDIT_DATA
            self.ui.buttonToggleMode.setChecked(False)
            self.romp.set_grid_mode(False)
        self.romp.grid_dirty = True # Force redraw to update handles
        self.display_image()

    ########################################
    # Slots for QActions from the UI       #
    ########################################

    @QtCore.pyqtSlot()
    def on_actionAbout_triggered(self):
        RomparAboutDialog.showAboutRompar(self)

    @QtCore.pyqtSlot()
    def on_actionManual_triggered(self):
        RomparAboutDialog.showAboutManual(self)

    @QtCore.pyqtSlot()
    def on_actionAuthors_triggered(self):
        RomparAboutDialog.showAboutAuthors(self)

    @QtCore.pyqtSlot()
    def on_actionLicense_triggered(self):
        RomparAboutDialog.showAboutLicense(self)

    @QtCore.pyqtSlot()
    def on_actionFindHex_triggered(self):
        data, okpressed = FindHexDialog.getBytes(self, self.romp.Search_HEX)
        if okpressed:
            self.showTempStatus('searching for',":".join((hex(b)[2:] for b in data)))
            if data == self.romp.Search_HEX:
                return
            self.romp.Search_HEX = data
            self.display_image()

    @QtCore.pyqtSlot()
    def on_actionSave_triggered(self):
        self.save_grid(backup=False)

    @QtCore.pyqtSlot()
    def on_actionBackupSave_triggered(self):
        self.save_grid(backup=True)

    @QtCore.pyqtSlot()
    def on_actionSaveAs_triggered(self):
        default_fn = self.grid_fn if self.grid_fn else self.romp.img_fn.parent
        name, filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Grid (json) File', str(default_fn), "Grid (*.json)")
        if (name, filter) == ('', ''):
            return
        old_grid_fn = self.grid_fn
        self.grid_fn = pathlib.Path(name).expanduser().absolute()
        if self.save_grid(backup=False):
            self.saven = 0
        else:
            self.grid_fn = old_grid_fn # Restore old value if save as failed.

    @QtCore.pyqtSlot()
    def on_actionSaveDataAsText_triggered(self):
        if self.grid_fn is not None:
            defname = str(self.grid_fn.with_suffix('.txt'))
        else:
            defname = ''
        fname, filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Data as txt file', defname , "Text (*.txt)")
        if (fname, filter) == ('', ''):
            return
        filepath = pathlib.Path(fname).expanduser().absolute()
        if self.save_data_as_text(filepath):
            self.showTempStatus('Exported data to', str(filepath))

    @QtCore.pyqtSlot()
    def on_actionUndo_triggered(self):
        if self.romp.history.undo():
            self.romp.grid_dirty = True
            self.display_image()
            self.showTempStatus('Undid last action')
        else:
             self.showTempStatus('Nothing to Undo')

    @QtCore.pyqtSlot()
    def on_actionRedo_triggered(self):
        if self.romp.history.redo():
            self.romp.grid_dirty = True
            self.display_image()
            self.showTempStatus('Redid last action')
        else:
             self.showTempStatus('Nothing to Redo')

    @QtCore.pyqtSlot()
    def on_actionRedrawGrid_triggered(self):
        self.romp.grid_dirty = True
        self.romp.redraw_grid()
        self.display_image()
        self.showTempStatus('Grid Redrawn')

    @QtCore.pyqtSlot()
    def on_actionRereadData_triggered(self):
        button = QtWidgets.QMessageBox.question(
            self, 'Re-read Data?',
            "Are you sure you want to reread the data? "
            "Any manual edits will be lost.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if button == QtWidgets.QMessageBox.Yes:
            self.romp.read_data()
            self.display_image()
            self.showTempStatus('Re-read data.')

    @QtCore.pyqtSlot()
    def on_actionToggleMode_triggered(self):
        self.ui.buttonToggleMode.setChecked(not self.ui.buttonToggleMode.isChecked())

    @QtCore.pyqtSlot(bool)
    def on_buttonToggleMode_toggled(self, checked):
        if checked:
            self.set_edit_mode(MODE_EDIT_GRID)
        else:
            self.set_edit_mode(MODE_EDIT_DATA)

    @QtCore.pyqtSlot()
    def on_actionGridEditMode_triggered(self):
        self.ui.buttonToggleMode.setChecked(True)

    @QtCore.pyqtSlot()
    def on_actionDataEditMode_triggered(self):
        self.ui.buttonToggleMode.setChecked(False)

    # Increment/Decrement values
    def get_view_rect(self):
        # Get visible area in scene coords
        viewport_rect = self.ui.graphicsView.viewport().rect()
        scene_rect = self.ui.graphicsView.mapToScene(viewport_rect).boundingRect()
        return (scene_rect.x(), scene_rect.y(), scene_rect.width(), scene_rect.height())

    @QtCore.pyqtSlot()
    def on_actionRadiusIncrease_triggered(self):
        self.config.radius += 1
        self.showTempStatus('Radius %d' % self.config.radius)
        self.romp.grid_dirty = True
        self.romp.read_data()
        self.display_image()
    @QtCore.pyqtSlot()
    def on_actionRadiusDecrease_triggered(self):
        self.config.radius = max(self.config.radius-1, 1)
        self.showTempStatus('Radius %d' % self.config.radius)
        self.romp.grid_dirty = True
        self.romp.read_data()
        self.display_image()

    @QtCore.pyqtSlot()
    def on_actionDilateIncrease_triggered(self):
        self.config.dilate += 1
        self.showTempStatus('Dilate %d' % self.config.dilate)
        self.romp.read_data()
        self.display_image()
    @QtCore.pyqtSlot()
    def on_actionDilateDecrease_triggered(self):
        self.config.dilate = max(self.config.dilate - 1, 0)
        self.showTempStatus('Dilate %d' % self.config.dilate)
        self.romp.read_data()
        self.display_image()

    @QtCore.pyqtSlot()
    def on_actionErodeIncrease_triggered(self):
        self.config.erode += 1
        self.showTempStatus('Erode %f' % self.config.erode)
        self.romp.read_data()
        self.display_image()
    @QtCore.pyqtSlot()
    def on_actionErodeDecrease_triggered(self):
        self.config.font_size = max(self.config.font_size - 0.1, 0)
        self.showTempStatus('Erode %f' % self.config.erode)
        self.romp.read_data()
        self.display_image()

    @QtCore.pyqtSlot()
    def on_actionFontIncrease_triggered(self):
        self.config.font_size += 0.1
        self.showTempStatus('Font Size %f' % self.config.font_size)
        self.romp.read_data()
        self.display_image()
    @QtCore.pyqtSlot()
    def on_actionFontDecrease_triggered(self):
        self.config.font_size = max(self.config.font_size - 0.1, 0)
        self.showTempStatus('Font Size %f' % self.config.font_size)
        self.romp.read_data()
        self.display_image()

    @QtCore.pyqtSlot()
    def on_actionBitThresholdDivisorIncrease_triggered(self):
        self.config.bit_thresh_div += 1
        self.showTempStatus('Threshold div %d' % self.config.bit_thresh_div)
        self.romp.read_data()
        self.display_image()
    @QtCore.pyqtSlot()
    def on_actionBitThresholdDivisorDecrease_triggered(self):
        self.config.bit_thresh_div -= 1
        self.showTempStatus('Threshold div %d' % self.config.bit_thresh_div)
        self.romp.read_data()
        self.display_image()

    @QtCore.pyqtSlot()
    def on_actionPixelThresholdMinimumIncrease_triggered(self):
        self.config.pix_thresh_min = min(self.config.pix_thresh_min + 1, 0xFF)
        self.showTempStatus('Threshold filter %d' % self.config.pix_thresh_min)
        self.romp.read_data()
        self.display_image()
    @QtCore.pyqtSlot()
    def on_actionPixelThresholdMinimumDecrease_triggered(self):
        self.config.pix_thresh_min = max(self.config.pix_thresh_min - 1, 0x01)
        self.showTempStatus('Threshold filter %d' % self.config.pix_thresh_min)
        self.romp.read_data()
        self.display_image()

    # Change the base image of the display.
    @QtCore.pyqtSlot()
    def on_actionImgBGBlank_triggered(self):
        self.showTempStatus('BG Image: Blank')
        self.config.img_display_blank_image = True
        self.config.img_display_original = False
        self.display_image()
    @QtCore.pyqtSlot()
    def on_actionImgBGOriginal_triggered(self):
        self.showTempStatus('BG Image: Original')
        self.config.img_display_blank_image = False
        self.config.img_display_original = True
        self.display_image()
    @QtCore.pyqtSlot()
    def on_actionImgBGTarget_triggered(self):
        self.showTempStatus('BG Image: Target')
        self.config.img_display_blank_image = False
        self.config.img_display_original = False
        self.display_image()

    # Toggle Options
    @QtCore.pyqtSlot(bool)
    def on_actionShowGrid_triggered(self, checked):
        self.showTempStatus('Display Grid', "on" if checked else "off")
        self.config.img_display_grid = checked
        self.display_image()

    @QtCore.pyqtSlot(bool)
    def on_actionShowDataBinary_triggered(self, checked):
        self.showTempStatus('Display Data in', "BIN" if checked else "HEX")
        self.config.img_display_binary = checked
        self.display_image()

    @QtCore.pyqtSlot(bool)
    def on_actionShowPeephole_triggered(self, checked):
        self.showTempStatus('Peephole Mask', "on" if checked else "off")
        self.config.img_display_peephole = checked
        self.display_image()

    @QtCore.pyqtSlot(bool)
    def on_actionShowData_triggered(self, checked):
        self.showTempStatus('Display Data', "on" if checked else "off")
        self.config.img_display_data = checked
        self.display_image()

    @QtCore.pyqtSlot(bool)
    def on_actionDataInverted_triggered(self, checked):
        self.showTempStatus('Display %sinverted' % ("" if checked else "NOT "))
        self.config.inverted = checked
        self.display_image()

    @QtCore.pyqtSlot(bool)
    def on_actionDataLSBitMode_triggered(self, checked):
        self.showTempStatus('Data', "LSB" if checked else "MSB")
        self.config.LSB_Mode = checked
        if self.config.img_display_data:
            self.display_image()

    # Edit Grid Buttons
    @QtCore.pyqtSlot()
    def on_actionDeleteColumn_triggered(self):
        if self.romp.Edit_x >= 0:
            if self.romp.del_bit_column(self.romp.Edit_x):
                self.romp.Edit_x, self.romp.Edit_y = -1, -1
                self.display_image()
                self.showTempStatus('Deleted Column')

    @QtCore.pyqtSlot()
    def on_actionDeleteRow_triggered(self):
        if self.romp.Edit_y >= 0:
            if self.romp.del_bit_row(self.romp.Edit_y):
                self.romp.Edit_x, self.romp.Edit_y = -1, -1
                self.display_image()
                self.showTempStatus('Deleted Row')

    @QtCore.pyqtSlot()
    def on_actionMoveColumnLeft_triggered(self):
        if self.romp.Edit_x >= 0:
            if self.romp.move_bit_column(self.romp.Edit_x, -1, relative=True):
                self.display_image()

    @QtCore.pyqtSlot()
    def on_actionMoveColumnRight_triggered(self):
        if self.romp.Edit_x >= 0:
            if self.romp.move_bit_column(self.romp.Edit_x, 1, relative=True):
                self.display_image()

    @QtCore.pyqtSlot()
    def on_actionMoveRowDown_triggered(self):
        if self.romp.Edit_y >= 0:
            if self.romp.move_bit_row(self.romp.Edit_y, 1, relative=True):
                self.display_image()

    @QtCore.pyqtSlot()
    def on_actionMoveRowUp_triggered(self):
        if self.romp.Edit_y >= 0:
            if self.romp.move_bit_row(self.romp.Edit_y, -1, relative=True):
                self.display_image()


    ############################################
    # Slots for Mouse Events from Graphicsview #
    ############################################

    @QtCore.pyqtSlot(QtCore.QPointF, int)
    def on_graphicsView_sceneLeftClicked(self, qimg_xy, keymods):
        img_xy = ImgXY(int(qimg_xy.x()), int(qimg_xy.y()))
        if self.mode == MODE_EDIT_DATA: # Data Edit Mode
            try:
                self.romp.toggle_data(self.romp.imgxy_to_bitxy(img_xy))
                self.display_image()
            except IndexError as e:
                print("No bit toggled")
        elif self.mode == MODE_EDIT_GRID: # Grid Edit Mode
            hit = self.romp.grid_hit_test(img_xy)
            if hit:
                 # Handled by eventFilter (Press/Release)
                 pass
            else:
                 do_autocenter = keymods & QtCore.Qt.ShiftModifier
                 self.romp.grid_add_vertical_line(img_xy, do_autocenter)
            self.display_image()
    @QtCore.pyqtSlot(QtCore.QPointF, int)
    def on_graphicsView_sceneRightClicked(self, qimg_xy, keymods):
        img_xy = ImgXY(int(qimg_xy.x()), int(qimg_xy.y()))
        if self.mode == MODE_EDIT_DATA: # Data Edit Mode
            self.select_bit_group(img_xy)
        elif self.mode == MODE_EDIT_GRID: # Grid Edit Mode
            do_autocenter = keymods & QtCore.Qt.ShiftModifier
            self.romp.grid_add_horizontal_line(img_xy, do_autocenter)
            self.display_image()

    def select_bit_group(self, img_xy):
        try:
            bit_xy = self.romp.imgxy_to_bitxy(img_xy)
            if bit_xy == (self.romp.Edit_x, self.romp.Edit_y):
                self.romp.Edit_x, self.romp.Edit_y = -1, -1
            else:
                self.romp.Edit_x, self.romp.Edit_y = bit_xy
            self.display_image()
            self.showTempStatus("Edit x,y:",
                                self.romp.Edit_x, self.romp.Edit_y)
        except IndexError as e:
            self.showTempStatus("No bit group selected")

    def eventFilter(self, source, event):
        if source == self.ui.graphicsView.viewport():
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if self.mode == MODE_EDIT_GRID and event.button() == QtCore.Qt.LeftButton:
                    # Map to scene
                    scene_pos = self.ui.graphicsView.mapToScene(event.pos())
                    img_xy = ImgXY(int(scene_pos.x()), int(scene_pos.y()))
                    
                    hit = self.romp.grid_hit_test(img_xy)
                    if hit:
                        idx, handle, is_vert = hit
                        self.dragging_handle = True
                        self.last_mouse_pos = scene_pos
                        self.drag_start_pos = scene_pos # Capture start position
                        
                        # Selection logic is also handled by signal, but setting it here ensures immediate drag response
                        self.romp.selected_handle = handle
                        is_ctrl = event.modifiers() & QtCore.Qt.ControlModifier
                        self.did_drag_select_add = False
                        
                        if is_vert:
                             self.drag_axis = 1
                             if is_ctrl:
                                  if idx not in self.romp.selected_indices_v:
                                      self.romp.select_toggle_v(idx)
                                      self.did_drag_select_add = True
                             else:
                                  if idx not in self.romp.selected_indices_v:
                                      self.romp.selected_line_v = idx
                                      self.romp.selected_line_h = None
                             
                             self.drag_start_selection = list(self.romp.selected_indices_v)
                             self.drag_start_index = idx
                        else:
                             self.drag_axis = 0
                             if is_ctrl:
                                  if idx not in self.romp.selected_indices_h:
                                      self.romp.select_toggle_h(idx)
                                      self.did_drag_select_add = True
                             else:
                                  if idx not in self.romp.selected_indices_h:
                                      self.romp.selected_line_v = None
                                      self.romp.selected_line_h = idx
                                      
                             self.drag_start_selection = list(self.romp.selected_indices_h)
                             self.drag_start_index = idx
                             
                             self.drag_start_selection = list(self.romp.selected_indices_h)
                             self.drag_start_index = idx
                             
                        # Create overlay items for dragged lines
                        self.drag_lines = []
                        indices = self.drag_start_selection if self.drag_start_selection else [idx]
                        lines = self.romp._grid_lines_v if self.drag_axis == 1 else self.romp._grid_lines_h
                        
                        for i in indices:
                            if 0 <= i < len(lines):
                                l = lines[i]
                                item = QtWidgets.QGraphicsLineItem()
                                pen = QtGui.QPen(QtCore.Qt.green)
                                pen.setWidth(2)
                                item.setPen(pen)
                                item.setZValue(101) # Above selection box
                                
                                if self.drag_axis == 1: # Vert
                                     item.setLine(l.start, 0, l.end, self.romp.img_height)
                                else: # Horiz
                                     item.setLine(0, l.start, self.romp.img_width, l.end)
                                
                                self.scene.addItem(item)
                                # Store item and original coordinates to compute offsets
                                self.drag_lines.append((item, l.start, l.end))

                        self.romp.grid_dirty = True
                        self.display_image()
                    else:
                        # Empty space click in Grid Mode
                        modifiers = event.modifiers()
                        if modifiers & QtCore.Qt.ShiftModifier:
                             # Shift-Click -> Start Box Selection
                             # Also blocking the creation of new lines by returning True
                             self.selecting_box = True
                             self.drag_start_pos_box = scene_pos
                             self.last_mouse_pos = scene_pos 
                             return True
                        
                        # Otherwise allow propagation (Add Line)
            
            elif event.type() == QtCore.QEvent.MouseMove:
                if self.dragging_handle and self.drag_start_pos:
                    scene_pos = self.ui.graphicsView.mapToScene(event.pos())
                    # Use total delta from start for overlay updates
                    total_dx = int(scene_pos.x() - self.drag_start_pos.x())
                    total_dy = int(scene_pos.y() - self.drag_start_pos.y())
                    
                    if total_dx != 0 or total_dy != 0:
                        # Update overlay items instead of full render
                        handle_type = self.romp.selected_handle
                        
                        for item, start, end in self.drag_lines:
                             new_start = start
                             new_end = end
                             
                             if self.drag_axis == 1: # Vertical
                                  if handle_type in ('start', 'both'): new_start += total_dx
                                  if handle_type in ('end', 'both'): new_end += total_dx
                                  item.setLine(new_start, 0, new_end, self.romp.img_height)
                             else: # Horizontal
                                  if handle_type in ('start', 'both'): new_start += total_dy
                                  if handle_type in ('end', 'both'): new_end += total_dy
                                  item.setLine(0, new_start, self.romp.img_width, new_end)
                        
                        self.last_mouse_pos = scene_pos
                        # self.display_image(fast=True) # REMOVED
                elif self.selecting_box and self.drag_start_pos_box:
                    # Update display with selection rect
                    scene_pos = self.ui.graphicsView.mapToScene(event.pos())
                    self.last_mouse_pos = scene_pos # Capture current end
                    
                    # Update QGraphicsRectItem
                    x = min(self.drag_start_pos_box.x(), scene_pos.x())
                    y = min(self.drag_start_pos_box.y(), scene_pos.y())
                    w = abs(scene_pos.x() - self.drag_start_pos_box.x())
                    h = abs(scene_pos.y() - self.drag_start_pos_box.y())
                    self.selection_box_item.setRect(x, y, w, h)
                    self.selection_box_item.show()
                    # No need to call display_image() here for selection box
            
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                if self.dragging_handle:
                    self.dragging_handle = False
                    
                    # calculate total delta
                    current_scene_pos = self.ui.graphicsView.mapToScene(event.pos())
                    # Use drag_start_pos if valid, else last_mouse_pos (fallback)
                    start_pos = self.drag_start_pos if self.drag_start_pos else self.last_mouse_pos
                    
                    total_dx = int(current_scene_pos.x() - start_pos.x())
                    total_dy = int(current_scene_pos.y() - start_pos.y())
                    
                    handle_type = self.romp.selected_handle

                    if self.drag_start_index is not None and (total_dx != 0 or total_dy != 0):
                        if self.drag_axis == 1: # Was moving column
                             indices = self.drag_start_selection if self.drag_start_selection else [self.drag_start_index]
                             # NOW apply the move to the model once
                             # Note: The model wasn't moved during drag, so we need to move it now.
                             # But `move_bit_column` usually takes one index (Edit_x). 
                             # If we have multiple indices, we need to be careful.
                             # `move_bit_column` implementation might only move one line? 
                             # Let's check `move_bit_column` logic. It moves `Edit_x` OR `indices`?
                             # In original code, it called `move_bit_column(self.romp.Edit_x ...)`.
                             # If multiselect, `Edit_x` is just the primary one. 
                             # Does `move_bit_column` handle multiselect?
                             # Looking at `rompar.py` (not visible here but implied/assumed), 
                             # `move_bit_column` seems to handle it if `idx` is passed.
                             # Actually, in original code `MouseMove`: `self.romp.move_bit_column(self.romp.Edit_x, dx...)`.
                             # So we should replicate that call here with `total_dx`.
                             self.romp.move_bit_column(self.romp.Edit_x, total_dx, relative=True, push_history=False)
                             
                             cmd = MoveColumnCommand(self.romp, self.drag_start_index, total_dx, relative=True, indices=indices, handle_type=handle_type)
                             # Where do I get final_idx/indices?
                             # I need to know where they ended up.
                             # They already moved (via MouseMove).
                             # So I don't need to execute, but I need final positions.
                             # _move logic returns final indices based on CURRENT state.
                             # But here state is already final.
                             # I need to "re-capture" indices?
                             # The indices might have shuffled.
                             # Since I sorted/rebuilt during drag, 'indices' (from before drag?) No.
                             # During drag, selected_indices are updated by sort_and_rebuild.
                             # So self.romp.selected_indices_v contains the FINAL indices.
                             cmd.final_indices = list(self.romp.selected_indices_v)
                             # And final_idx?
                             cmd.final_idx = self.romp.Edit_x
                             self.romp.history.push(cmd)
                             
                        elif self.drag_axis == 0: # Was moving row
                             indices = self.drag_start_selection if self.drag_start_selection else [self.drag_start_index]
                             # NOW apply the move to the model once
                             self.romp.move_bit_row(self.romp.Edit_y, total_dy, relative=True, push_history=False)
                             
                             cmd = MoveRowCommand(self.romp, self.drag_start_index, total_dy, relative=True, indices=indices, handle_type=handle_type)
                             cmd.final_indices = list(self.romp.selected_indices_h)
                             cmd.final_idx = self.romp.Edit_y
                             self.romp.history.push(cmd)



                    elif self.drag_start_index is not None:
                         # Click logic (No drag)
                         is_ctrl = event.modifiers() & QtCore.Qt.ControlModifier
                         if is_ctrl:
                              if not self.did_drag_select_add:
                                   if self.drag_axis == 1:
                                       self.romp.select_toggle_v(self.drag_start_index)
                                   else:
                                       self.romp.select_toggle_h(self.drag_start_index)
                         else:
                              # Exclusive 
                              if self.drag_axis == 1:
                                   self.romp.selected_line_v = self.drag_start_index
                                   self.romp.selected_line_h = None
                              else:
                                   self.romp.selected_line_v = None
                                   self.romp.selected_line_h = self.drag_start_index

                    # Cleanup drag items
                    for item, _, _ in self.drag_lines:
                         self.scene.removeItem(item)
                    self.drag_lines = []

                    self.last_mouse_pos = None
                    self.drag_start_pos = None
                    self.drag_start_index = None
                    self.drag_axis = None
                    self.did_drag_select_add = False

                    self.romp.grid_dirty = True
                    self.display_image(fast=False)
                
                elif self.selecting_box:
                    self.selecting_box = False
                    current_scene_pos = self.ui.graphicsView.mapToScene(event.pos())
                    start_pos = self.drag_start_pos_box
                    
                    if start_pos:
                        x = min(start_pos.x(), current_scene_pos.x())
                        y = min(start_pos.y(), current_scene_pos.y())
                        w = abs(current_scene_pos.x() - start_pos.x())
                        h = abs(current_scene_pos.y() - start_pos.y())
                        rect = (int(x), int(y), int(w), int(h))
                        
                        is_ctrl = event.modifiers() & QtCore.Qt.ControlModifier
                        if self.romp.select_in_rect(rect, add=is_ctrl):
                            pass # Display updated by display_image
                    
                    self.drag_start_pos_box = None
                    self.selection_box_item.hide()
                    self.display_image(fast=False)

        return super(RomparUiQt, self).eventFilter(source, event)

def load_anotate(fn):
    """
    Really this is a set of (row, col): how, but that type of key doesn't map well to json
    So instead re-process the keys
    In: "1,2"
    Out: (1, 2)

    Ex: annotate col=1, row=2 red
    {
        "1,2": {"color": [255, 0, 0]}
    }
    """
    j = json_load_exit_bad(fn, "--annotate")

    ret = {}
    for k, v in j.items():
        c,r = k.split(",")
        ret[(int(c), int(r))] = v


def run(app):
    import argparse

    if len(sys.argv) <= 1:
        exit_message("Arguments required, try --help", prefer_cli=None)

    parser = argparse.ArgumentParser(description='Extract mask ROM image')
    parser.add_argument('--radius', type=int,
                        help='Use given radius for display, '
                        'bounded square for detection')
    parser.add_argument('--bit-thresh-div', type=str,
                        help='Bit set area threshold divisor')
    # Only care about min
    parser.add_argument('--pix-thresh', type=str,
                        help='Pixel is set threshold minimum')
    parser.add_argument('--dilate', type=str, help='Dilation')
    parser.add_argument('--erode', type=str, help='Erosion')
    parser.add_argument('--debug', action='store_true', help='')
    parser.add_argument('--load', help='Load saved grid file')
    parser.add_argument('--txt', help='Load given .txt instead of .json binary')
    parser.add_argument('--dx', type=int, help='Shift data relative to image x pixels')
    parser.add_argument('--dy', type=int, help='Shift data relative to image y pixels')
    parser.add_argument('--annotate', help='Annotation .json')
    parser.add_argument('image', nargs='?', help='Input image')
    parser.add_argument('cols_per_group', nargs='?', type=int, help='')
    parser.add_argument('rows_per_group', nargs='?', type=int, help='')
    parser.add_argument('-j', '--threads', type=int, default=4, help='Number of threads for grid redraw')
    args = parser.parse_args()

    config = Config()
    if args.threads:
        config.threads = args.threads
    if args.radius:
        config.default_radius = args.radius
        config.radius = args.radius
    if args.bit_thresh_div:
        config.bit_thresh_div = int(args.bit_thresh_div, 0)
    if args.pix_thresh:
        config.pix_thresh_min = int(args.pix_thresh, 0)
    if args.dilate:
        config.dilate = int(args.dilate, 0)
    if args.erode:
        config.erode = int(args.erode, 0)
    annotate = None


    if args.annotate:
        annotate = load_anotate(args.annotate)

    window = RomparUiQt(config,
                        img_fn=args.image, grid_fn=args.load,
                        group_cols=args.cols_per_group,
                        group_rows=args.rows_per_group,
                        txt=args.txt, annotate=annotate)
    if args.dx or args.dy:
        window.shift_xy(args.dx, args.dy)
    window.show()

    return app.exec_() # Start the event loop.

def main():
    import sys

    # Initialize the QApplication object, and free it last.
    # Not having this in a different function than other QT
    # objects can cause segmentation faults as app is freed
    # before the QEidgets.
    app = QtWidgets.QApplication(sys.argv)

    # Allow Ctrl-C to interrupt QT by scheduling GIL unlocks.
    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None) # Let the interpreter run.

    sys.exit(run(app))

if __name__ == "__main__":
    main()
