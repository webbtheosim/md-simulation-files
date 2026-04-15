#!/bin/bash

# Source and destination paths
SRC_PATH="/scratch/gpfs/WEBB/jv6139/M_monomer/control-scripts/tREM_GPU/rem_template.mdp"
DEST_DIR="/scratch/gpfs/WEBB/jv6139/M_monomer/control-scripts/tREM_GPU"

# List of temperatures
#TEMPERATURES=(260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440)
TEMPERATURES=(260 262.5 265 267.5 270 272.5 275 277.5 280 282.5 285 287.5 290 292.5 295 297.5 300 302.5 305 307.5 310 312.5 315 317.5 320 322.5 325 327.5 330 332.5 335 337.5 340 342.5 345 347.5 350 352.5 355 357.5 360 362.5 365 367.5 370 372.5 375 377.5 380 382.5 385 387.5 390 392.5 395 397.5 400 402.5 405 407.5 410 412.5 415 417.5 420 422.5 425 427.5 430 432.5 435 437.5 440)
# Loop through each temperature
for TEMP in "${TEMPERATURES[@]}"; do
    # Copy the template file to the destination directory
    cp "${SRC_PATH}" "${DEST_DIR}/trem_${TEMP}.mdp"
    # Replace SET_TEMP in the copied file
    sed -i "s/SET_TEMP/${TEMP}/g" "${DEST_DIR}/trem_${TEMP}.mdp"
    echo "Created ${DEST_DIR}/trem_${TEMP}.mdp with temperature ${TEMP}"
done