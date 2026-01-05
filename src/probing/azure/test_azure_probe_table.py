import time
import datetime
import threading
from supabase import create_client, Client
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute.models import (
    VirtualMachine,
    HardwareProfile,
    NetworkProfile,
    OSProfile,
    LinuxConfiguration,
    SshConfiguration,
    SshPublicKey,
    StorageProfile,
    ImageReference,
    OSDisk,
    DiskCreateOptionTypes,
    NetworkInterfaceReference,
)

# ========================
# Configuration Parameters
# ========================
SUBSCRIPTION_ID = "<your-azure-subscription-id>"
RESOURCE_GROUP = "<your-resource-group>"
LOCATION = "eastus"
VM_SIZE = "Standard_B1s"  # Smallest VM type for test
SSH_KEY_PATH = "<your-ssh-public-key-path>"

probe_interval_minutes = 5
max_probe_runtime_minutes = 2
status_check_interval_seconds = 60
instances_per_interval = 1

# ========================
# Azure Clients
# ========================
credential = DefaultAzureCredential()
compute_client = ComputeManagementClient(credential, SUBSCRIPTION_ID)

# ========================
# Supabase Connection
# ========================
SUPABASE_URL = "https://bdrsaepidwmokgyzmfuk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJkcnNhZXBpZHdtb2tneXptZnVrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg0MDk3MTAsImV4cCI6MjA3Mzk4NTcxMH0.DAnzJQfaMBO_RdTJWDtywPVRVm3kZzsoRPoYq_SUNig"   
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def log_probe_result(instance_id, start_time, end_time, duration, outcome):
    """Insert probe result into Supabase."""
    data = {
        "instance_id": instance_id if instance_id else "LaunchFailed",
        "region": LOCATION,
        "instance_type": VM_SIZE,
        "start_time_utc": start_time.isoformat() if start_time else None,
        "end_time_utc": end_time.isoformat() if end_time else None,
        "duration_minutes": duration,
        "outcome": outcome
    }
    supabase.table("probe_results").insert(data).execute()
    print(f"Logged result for {instance_id if instance_id else 'LaunchFailed'} to Supabase.")

# ========================
# Probe Functions
# ========================
def launch_spot_probes(requested_count):
    """Launch multiple Spot VMs, return successful launches and failures."""
    launched = []
    failures = 0

    for i in range(requested_count):
        vm_name = f"spot-probe-{int(time.time())}-{i}"
        try:
            vm_params = {
                "location": LOCATION,
                "priority": "Spot",
                "eviction_policy": "Deallocate",  # or Delete
                "hardware_profile": HardwareProfile(vm_size=VM_SIZE),
                "storage_profile": StorageProfile(
                    image_reference=ImageReference(
                        publisher="Canonical",
                        offer="UbuntuServer",
                        sku="18.04-LTS",
                        version="latest"
                    ),
                    os_disk=OSDisk(
                        create_option=DiskCreateOptionTypes.from_image,
                        name=f"{vm_name}-osdisk"
                    )
                ),
                "os_profile": OSProfile(
                    computer_name=vm_name,
                    admin_username="azureuser",
                    linux_configuration=LinuxConfiguration(
                        disable_password_authentication=True,
                        ssh=SshConfiguration(
                            public_keys=[SshPublicKey(
                                path=f"/home/azureuser/.ssh/authorized_keys",
                                key_data=open(SSH_KEY_PATH).read()
                            )]
                        )
                    )
                ),
                "network_profile": NetworkProfile(
                    network_interfaces=[NetworkInterfaceReference(
                        id=f"/subscriptions/{SUBSCRIPTION_ID}/resourceGroups/{RESOURCE_GROUP}/providers/Microsoft.Network/networkInterfaces/{vm_name}-nic"
                    )]
                )
            }

            async_vm = compute_client.virtual_machines.begin_create_or_update(
                RESOURCE_GROUP, vm_name, vm_params
            )
            async_vm.result()  # Wait for completion
            start_time = datetime.datetime.utcnow()
            launched.append((vm_name, start_time))
            print(f"[{start_time}] Launched probe VM {vm_name}")
        except Exception as e:
            print(f"Launch failed for {vm_name}: {e}")
            failures += 1

    return launched, failures

def monitor_spot_probe(vm_name, start_time):
    """Monitor a Spot VM until eviction or max runtime."""
    elapsed_minutes = 0

    while elapsed_minutes < max_probe_runtime_minutes:
        try:
            instance_view = compute_client.virtual_machines.instance_view(
                RESOURCE_GROUP, vm_name
            )
            statuses = [s.code for s in instance_view.statuses]
            if any("PowerState/deallocated" in s for s in statuses):
                end_time = datetime.datetime.utcnow()
                duration = (end_time - start_time).total_seconds() / 60
                print(f"[{end_time}] {vm_name} evicted after {duration:.1f} minutes")
                return end_time, duration, "Interrupted"
        except Exception as e:
            print(f"Error checking VM {vm_name}: {e}")
            break

        time.sleep(status_check_interval_seconds)
        elapsed_minutes += status_check_interval_seconds / 60

    # If VM survived horizon → delete
    end_time = datetime.datetime.utcnow()
    try:
        compute_client.virtual_machines.begin_delete(RESOURCE_GROUP, vm_name).result()
    except Exception as e:
        print(f"Error deleting {vm_name}: {e}")
    duration = (end_time - start_time).total_seconds() / 60
    print(f"[{end_time}] {vm_name} survived horizon ({duration:.1f} min), terminated manually")
    return end_time, duration, "Censored (Max Runtime Reached)"

# ========================
# Worker Thread Wrapper
# ========================
def monitor_and_log(vm_name, start_time):
    """Thread target: monitor a VM and log result."""
    end_time, duration, outcome = monitor_spot_probe(vm_name, start_time)
    log_probe_result(vm_name, start_time, end_time, duration, outcome)

# ========================
# Main Loop
# ========================
def run_probe_scheduler():
    while True:
        print(f"Requesting {instances_per_interval} Spot VMs...")
        launched_instances, failures = launch_spot_probes(instances_per_interval)

        for vm_name, start_time in launched_instances:
            t = threading.Thread(target=monitor_and_log, args=(vm_name, start_time))
            t.daemon = True
            t.start()

        if failures > 0:
            now = datetime.datetime.utcnow()
            for _ in range(failures):
                log_probe_result(None, now, now, 0.0, "LaunchFailed")

        print(f"Sleeping {probe_interval_minutes} minutes until next probe batch...")
        time.sleep(probe_interval_minutes * 60)

if __name__ == "__main__":
    run_probe_scheduler()
