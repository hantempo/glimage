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

    def test_rgb565_to_rgb8(self):
        input_pixels = np.array([0xF800, 0x07E0, 0x001F], dtype=np.uint16)
        expected_output_pixels = np.array([0xFF, 0x00, 0x00,
                                           0x00, 0xFF, 0x00,
                                           0x00, 0x00, 0xFF], dtype=np.uint8)
        output_pixels = np.empty(shape=(9,), dtype=np.uint8)
        self.assertTrue(_glimage.rgb565_to_rgb8(3, 1, input_pixels, output_pixels))
        self.assertTrue(np.array_equal(output_pixels, expected_output_pixels))

if __name__ == '__main__':
    unittest.main()