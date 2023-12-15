registered_device_stat = 0 
# Determine status based on RSSI value and UUID
if -100 <= dev.rssi <= -1:
    uuids = [value for (adtype, desc, value) in dev.getScanData() if adtype == 7]
    if uuids == ['42afc56a-8041-4cdd-bd24-04d58976fb2e']:
        status = "Enter"
        if status == last_status:
            status_counts["Enter"] = 0
            registered_device_stat =1 
            if registered_device_stat
        else:
            status_counts["Enter"] = 1
            ble = 1 if status == "Enter" else 0
    else:
        status_counts["Enter"] = 0
        ble = 0
    last_status = status