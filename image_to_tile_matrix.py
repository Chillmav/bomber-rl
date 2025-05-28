import numpy as np


def image_to_tile_matrix(image, cols=13, rows=11, tile_width=10, tile_height=10):
    
    tile_matrix = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            
            y1 = row * tile_height
            y2 = y1 + tile_height
            x1 = col * tile_width
            x2 = x1 + tile_width

            tile = image[y1:y2, x1:x2]
            avg_b = np.mean(tile[:,:,0])
            avg_g = np.mean(tile[:,:,1])
            avg_r = np.mean(tile[:,:,2])
            avg_value = np.mean(tile)
            
            if avg_value < 39:
                tile_matrix[row, col] = 2 # bricks
            elif avg_value == 74:
                tile_matrix[row, col] = 0 # path
            elif avg_value > 50 and avg_b > 30:
                tile_matrix[row, col] = 3 # player
            elif avg_b < 3:
                tile_matrix[row, col] = 0
            elif avg_b < 15:
                if tile_matrix[row, col - 1] > avg_b and tile_matrix[row, col - 1] not in [0, 1, 2, 3]:
                    tile_matrix[row, col] = 0
                elif tile_matrix[row - 1, col] > avg_b and tile_matrix[row - 1, col] not in [0, 1, 2, 3]:
                    tile[row, col] = 0
                else:
                    tile_matrix[row, col] = 4 # mob
    tile_matrix[1::2, 1::2] = 1 
    return tile_matrix