"""Sub-region tools:
    Tools are designed for icegame that scales up lattice size
    without re-training the CNN model.
"""

import numpy as np

def move_center(center, global_pos, lx, ly, Lx, Ly):
  """Move center according to relative poistion with box and global boundary.
  """
  pdb = lambda p, d, l: ((p+d)%l+l)%l
  x, y = center
  gX, gY = global_pos
  lx2 = lx //2
  ly2 = ly //2
  Lx2 = Lx //2
  Ly2 = Ly //2

  # outward vector
  dx = gX-x
  dy = gY-y
  adx = abs(dx)
  ady = abs(dy)

  if (adx <=lx2) and (ady <=ly2):
    return center
  else:
    # following are moving cases
    # 1. touch box boundary, one step increment
    # 2. cross global boundary,
    new_x, new_y = -1, -1

    # move x
    if (adx > 0):
      inc_x = dx-lx2 if dx > 0 else dx+lx2
      # check inc_x about boundary crossing case
      if (abs(inc_x) < Lx2):
        new_x = x + inc_x
      else:
        # crossing case
        virt_gX = gX+Lx if (inc_x<0) else gX-Lx
        virt_dx = virt_gX-x
        inc_x = virt_dx-lx2 if virt_dx > 0 else virt_dx+lx2
        new_x = pdb(x, inc_x, Lx)
    else:
      new_x = x

    # move y
    if (ady > 0):
      inc_y = dy-ly2 if dy > 0 else dy+ly2
      if (abs(inc_y) < Ly2):
        new_y = y + inc_y
      else:
        # crossing case
        virt_gY = gY+Ly if (inc_y<0) else gY-Ly
        virt_dy = virt_gY-y
        inc_y = virt_dy-ly2 if virt_dy > 0 else virt_dy+ly2
        new_y = pdb(y, inc_y, Ly)
    else:
      new_y = y

    # sanity check of negative position is needed.
    return (new_x, new_y)

def periodic_crop(img, center, lx, ly):
  """Crop the image with given center with periodic boundary condition.

    Periodic Boundary:
      check whether start & end exceeds boundary

    General algorithm
    if start < 0:
      rest of slice rounding.
      np.concat(img[new_start:None], img[0:end])
    else if end > L:
      round to starting index.
      np.concat(img[0:new_end], img[head:None])
    else:
      normal cropping, img[start:end]

    9 cases:
      * boundary_concat x4
      * corner_concat x4
      * normal
    Truth table:
    xs  xe  ys  ye
    --------------
    x   o   o   o
    o   x   o   o
    o   o   x   o
    o   o   o   x
    x   o   x   o
    o   x   x   o
    x   o   o   x
    o   x   o   x
    o   o   o   o
  """
  dim = len(img.shape)
  if dim == 2:
    Lx, Ly = img.shape
  elif dim == 3:
    Lx, Ly, _ = img.shape

  x, y = center

  # normal case:
  start_x = x - (lx//2)
  start_y = y - (ly//2)
  end_x = start_x+lx
  end_y = start_y+ly

  #print ("xs: {}, xe: {}, ys: {}, ye: {}".format(start_x, end_x, start_y, end_y))
  # sanity check
  if (end_x < start_x):
    print ("End_x is earlier than start_x, swap them")
    start_x, end_x = end_x, start_x
  elif (start_x == end_x):
    print ("Vanish x axis!")
  if (end_y < start_y):
    print ("End_y is eariler than start_y, swap them")
    start_y, end_y = end_y, start_y
  elif (start_y == end_y):
    print ("Vanish y axis!")

  cropped_img=None

  if (start_x <0 and end_x <=Lx) and (start_y >=0 and end_y <=Ly):
    # left concat
    new_start_x = start_x + Lx
    left = img[start_y:end_y, new_start_x:None]
    right = img[start_y:end_y, 0:end_x]
    cropped_img = np.concatenate([left, right], axis=1)
  elif (start_x >=0 and end_x >Lx) and (start_y >=0 and end_y <=Ly):
    # right concat
    new_end_x = end_x - Lx
    right = img[start_y:end_y, 0:new_end_x]
    left = img[start_y:end_y, start_x:None]
    cropped_img = np.concatenate([left, right], axis=1)
  elif (start_x >=0 and end_x <=Lx) and (start_y <0 and end_y <=Ly):
    # up concat
    new_start_y = start_y + Lx
    up = img[new_start_y:None, start_x:end_x]
    down = img[0:end_y, start_x:end_x]
    cropped_img = np.concatenate([up, down], axis=0)
  elif (start_x >=0 and end_x <=Lx) and (start_y >=0 and end_y >Ly):
    # down concat
    new_end_y = end_y - Ly
    down = img[0:new_end_y, start_x:end_x]
    up = img[start_y:None, start_x:end_x]
    cropped_img = np.concatenate([up, down], axis=0)
  elif(start_x <0 and end_x <=Lx) and (start_y <0 and end_y <=Ly):
    new_start_x = start_x + Lx
    new_start_y = start_y + Lx
    uL = img[0:end_y, 0:end_x]
    uR = img[0:end_y, new_start_x:None]
    dL = img[new_start_y:None, 0:end_x]
    dR = img[new_start_y:None, new_start_x:None]
    up = np.concatenate([dR, dL], axis=1)
    down = np.concatenate([uR, uL], axis=1)
    cropped_img = np.concatenate([up, down], axis=0)
  elif(start_x >=0 and end_x >Lx) and (start_y <0 and end_y <=Ly):
    new_end_x = end_x - Lx
    new_start_y = start_y + Lx
    uL = img[0:end_y, 0:new_end_x]
    uR = img[0:end_y, start_x:None]
    dR = img[new_start_y:None, start_x:None]
    dL = img[new_start_y:None, 0:new_end_x]
    up = np.concatenate([dR, dL], axis=1)
    down = np.concatenate([uR, uL], axis=1)
    cropped_img = np.concatenate([up, down], axis=0)
  elif(start_x <0 and end_x <=Lx) and (start_y >=0 and end_y >Ly):
    new_start_x = start_x + Lx
    new_end_y = end_y - Ly
    uL = img[0:new_end_y, 0:end_x]
    uR = img[0:new_end_y, new_start_x:None]
    dL = img[start_y:None, 0:end_x]
    dR = img[start_y:None, new_start_x:None]
    up = np.concatenate([dR, dL], axis=1)
    down = np.concatenate([uR, uL], axis=1)
    cropped_img = np.concatenate([up, down], axis=0)
  elif(start_x >=0 and end_x >Lx) and (start_y >=0 and end_y >Ly):
    new_end_x = end_x - Lx
    new_end_y = end_y - Ly
    dR = img[start_y:None, start_x:None]
    dL = img[start_y:None, 0:new_end_x]
    uR = img[0:new_end_y, start_x:None]
    uL = img[0:new_end_y, 0:new_end_x]
    up = np.concatenate([dR, dL], axis=1)
    down = np.concatenate([uR, uL], axis=1)
    cropped_img = np.concatenate([up, down], axis=0)
  else:
    cropped_img = img[start_y:end_y, start_x:end_x]

  return cropped_img


def show_region(img, cropped, center, lx, ly):
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  fig = plt.figure()

  x, y = center
  left = x - lx//2
  down = y - ly//2

  ax1 = plt.subplot(121)
  rect = patches.Rectangle((left, down), lx, ly, linewidth=2,edgecolor='r',facecolor='none')
  plt.imshow(img, 'gray', interpolation='None', aspect='equal') #,vmin=0, vmax=2*lx, cmap='jet', aspect='equal')
  ax1.add_patch(rect)

  plt.subplot(122)
  plt.imshow(cropped, 'gray', interpolation='None', aspect='equal')#,vmin=0, vmax=2*lx, cmap='jet',aspect='equal')
  plt.tight_layout()
  plt.show()
