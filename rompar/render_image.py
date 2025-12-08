    def render_image(self, img_display=None, rgb=False, viewport=None, fast=False):
        t = time.time()
        if img_display is None:
             img_display = numpy.zeros(self.img_original.shape, numpy.uint8)

        if self.config.img_display_blank_image:
            img_display.fill(0)
        elif self.config.img_display_original:
            numpy.copyto(img_display, self.img_original)
        else:
            numpy.copyto(img_display, self.img_target)

        if self.config.img_display_grid:
            self.redraw_grid(viewport=viewport, fast=fast)
            cv.bitwise_or(img_display, self.img_grid, img_display)

        if self.config.img_display_peephole:
            cv.bitwise_and(img_display, self.img_peephole, img_display)

        if self.config.img_display_data:
            self.render_data_layer(img_display)

        if self.annotate:
            self.render_annotate(img_display)

        # print("render_image time:", time.time()-t)

        if rgb:
            cv.cvtColor(img_display, cv.COLOR_BGR2RGB, img_display)

        return img_display
