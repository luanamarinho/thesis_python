# -------------------------------------------------------------------------
#
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
# ----------------------------------------------------------------------------------
# The example companies, organizations, products, domain names,
# e-mail addresses, logos, people, places, and events depicted
# herein are fictitious. No association with any real company,
# organization, product, domain name, email address, logo, person,
# places, or events is intended or should be inferred.
# --------------------------------------------------------------------------

# Global constant variables (Azure Storage account/Batch details)

# import "config.py" in "python_quickstart_client.py "
# Please note that storing the batch and storage account keys in Azure Key Vault
# is a better practice for Production usage.

"""
Configure Batch and Storage Account credentials
"""

BATCH_ACCOUNT_NAME = 'thesis'  # Your batch account name
BATCH_ACCOUNT_KEY = 'atTF/rd0NR0y1rQSqk5y5AGcdz4Nk+oEk105bDOkhW+dZLO57p/Vs7+bffRucetBHwlQkX3xu1p4+ABaTk1FPg=='  # Your batch account key '<primary access key>'
BATCH_ACCOUNT_URL = 'https://thesis.westeurope.batch.azure.com'  # Your batch account URL '<account endpoint>'
STORAGE_ACCOUNT_NAME = 'csb100320037b715e57' #'<storage account name>'
STORAGE_ACCOUNT_KEY = 'wgyMTU00S/XZultsTizn/SmKIzATx2T+Yjbw/ehd360a0F9yH5IB9BPLWxC4nNW0Ql5FLZgr1Qo++ASt8i3Gtw==' #'<key1>'
STORAGE_ACCOUNT_DOMAIN = 'blob.core.windows.net' # Your storage account blob service domain


POOL_ID = 'myPool'  # Your Pool ID
#POOL_NODE_COUNT = 2  # Pool node count
#POOL_VM_SIZE = 'STANDARD_DS1_V2'  # VM Type/Size
JOB_ID = 'PythonQuickstartJob'  # Job ID
#STANDARD_OUT_FILE_NAME = 'stdout.txt'  # Standard Output file
