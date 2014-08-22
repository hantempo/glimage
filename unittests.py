import unittest
import numpy as np

import _glimage

class TestGLImage(unittest.TestCase):

    def test_rgba4_to_rgba8(self):
        input_pixels = np.array([0xF000, 0x0F00, 0x00F0, 0x000F], dtype=np.uint16)
        expected_output_pixels = np.array([0xFF, 0x00, 0x00, 0x00,
                                  0x00, 0xFF, 0x00, 0x00,
                                  0x00, 0x00, 0xFF, 0x00,
                                  0x00, 0x00, 0x00, 0xFF], dtype=np.uint8)
        output_pixels = np.empty(shape=(16,), dtype=np.uint8)
        self.assertTrue(_glimage.rgba4_to_rgba8(2, 2, input_pixels, output_pixels))
        self.assertTrue(np.array_equal(output_pixels, expected_output_pixels))

    def test_rgb565_to_rgba8(self):
        input_pixels = np.array([0xF800, 0x07E0, 0x001F], dtype=np.uint16)
        expected_output_pixels = np.array([0xFF, 0x00, 0x00, 0xFF,
                                           0x00, 0xFF, 0x00, 0xFF,
                                           0x00, 0x00, 0xFF, 0xFF], dtype=np.uint8)
        output_pixels = np.empty(shape=(12,), dtype=np.uint8)
        self.assertTrue(_glimage.rgb565_to_rgba8(3, 1, input_pixels, output_pixels))
        self.assertTrue(np.array_equal(output_pixels, expected_output_pixels))

    def test_rgba5551_to_rgba8(self):
        input_pixels = np.array([0x1020, 0x3040, 0x5060], dtype=np.uint16)
        expected_output_pixels = np.array([0x10, 0x00, 0x83, 0x00,
                                  0x31, 0x08, 0x00, 0x00,
                                  0x52, 0x08, 0x83, 0x00], dtype=np.uint8)
        output_pixels = np.empty(shape=(12,), dtype=np.uint8)
        self.assertTrue(_glimage.rgba5551_to_rgba8(3, 1, input_pixels, output_pixels))
        self.assertTrue(np.array_equal(output_pixels, expected_output_pixels))

    def test_luminance_alpha_to_rgba8(self):
        input_pixels = np.array([0x10, 0x20, 0x30, 0x40, 0x50, 0x60], dtype=np.uint8)
        expected_output_pixels = np.array([0x10, 0x10, 0x10, 0x20,
                                  0x30, 0x30, 0x30, 0x40,
                                  0x50, 0x50, 0x50, 0x60], dtype=np.uint8)
        output_pixels = np.empty(shape=(12,), dtype=np.uint8)
        self.assertTrue(_glimage.luminance_alpha_to_rgba8(3, 1, input_pixels, output_pixels))
        self.assertTrue(np.array_equal(output_pixels, expected_output_pixels))

    def test_alpha_to_rgba8(self):
        input_pixels = np.array([0x10, 0x30, 0x50], dtype=np.uint8)
        expected_output_pixels = np.array([0x00, 0x00, 0x00, 0x10,
                                  0x00, 0x00, 0x00, 0x30,
                                  0x00, 0x00, 0x00, 0x50], dtype=np.uint8)
        output_pixels = np.empty(shape=(12,), dtype=np.uint8)
        self.assertTrue(_glimage.alpha_to_rgba8(3, 1, input_pixels, output_pixels))
        self.assertTrue(np.array_equal(output_pixels, expected_output_pixels))

    def test_luminance_to_rgba8(self):
        input_pixels = np.array([0x10, 0x30, 0x50], dtype=np.uint8)
        expected_output_pixels = np.array([0x10, 0x10, 0x10, 0xFF,
                                  0x30, 0x30, 0x30, 0xFF,
                                  0x50, 0x50, 0x50, 0xFF], dtype=np.uint8)
        output_pixels = np.empty(shape=(12,), dtype=np.uint8)
        self.assertTrue(_glimage.luminance_to_rgba8(3, 1, input_pixels, output_pixels))
        self.assertTrue(np.array_equal(output_pixels, expected_output_pixels))

    def test_R8_to_rgba8(self):
        input_pixels = np.array([0x10, 0x30, 0x50], dtype=np.uint8)
        expected_output_pixels = np.array([0x10, 0x00, 0x00, 0xFF,
                                  0x30, 0x00, 0x00, 0xFF,
                                  0x50, 0x00, 0x00, 0xFF], dtype=np.uint8)
        output_pixels = np.empty(shape=(12,), dtype=np.uint8)
        self.assertTrue(_glimage.r8_to_rgba8(3, 1, input_pixels, output_pixels))
        self.assertTrue(np.array_equal(output_pixels, expected_output_pixels))

    def test_rgb8_and_rgba8(self):
        input_pixels = np.array([0x10, 0x30, 0x50, 0x70, 0x90, 0xB0], dtype=np.uint8)
	expected_output_pixels = np.array([0x10, 0x30, 0x50, 0xFF,
            0x70, 0x90, 0xB0, 0xFF], dtype=np.uint8)
        output_pixels = np.empty(shape=(8,), dtype=np.uint8)
        self.assertTrue(_glimage.rgb8_to_rgba8(2, 1, input_pixels, output_pixels))
        self.assertTrue(np.array_equal(output_pixels, expected_output_pixels))
        output_pixels = np.empty(shape=(6,), dtype=np.uint8)
        self.assertTrue(_glimage.rgba8_to_rgb8(2, 1, expected_output_pixels, output_pixels))
        self.assertTrue(np.array_equal(output_pixels, input_pixels))

if __name__ == '__main__':
    unittest.main()
