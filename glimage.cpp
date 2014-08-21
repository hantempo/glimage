#include "glimage.h"

bool rgb565_to_rgb8(unsigned int width, unsigned int height, unsigned short *input_pixels, unsigned char *output_pixels)
{
    for (unsigned int index = 0; index < width * height; ++index)
    {
        const unsigned short pixel = *(input_pixels++);
        *(output_pixels++) = ((pixel >> 11) & 0x1F) * 0xFF / 0x1F;
        *(output_pixels++) = ((pixel >> 5) & 0x3F) * 0xFF / 0x3F;
        *(output_pixels++) = (pixel & 0x1F) * 0xFF / 0x1F;
    }
    return true;
}

bool rgba4_to_rgba8(unsigned int width, unsigned int height, unsigned short *input_pixels, unsigned char *output_pixels)
{
    for (unsigned int index = 0; index < width * height; ++index)
    {
        const unsigned short pixel = *(input_pixels++);
        *(output_pixels++) = ((pixel >> 12) & 0x0F) * 0xFF / 0x0F;
        *(output_pixels++) = ((pixel >> 8) & 0x0F) * 0xFF / 0x0F;
        *(output_pixels++) = ((pixel >> 4) & 0x0F) * 0xFF / 0x0F;
        *(output_pixels++) = (pixel & 0x0F) * 0xFF / 0x0F;
    }
    return true;
}