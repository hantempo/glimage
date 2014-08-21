#include "glimage.h"

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