# Databricks notebook source
# MAGIC %pip install --upgrade --quiet databricks-sdk mlflow delta-spark
# MAGIC %restart_python

# COMMAND ----------

from typing import Sequence

from importlib.metadata import version

pip_requirements: Sequence[str] = (
  f"databricks-sdk=={version('databricks-sdk')}",
  f"delta-spark=={version('delta-spark')}",
  f"mlflow=={version('mlflow')}",
)
print("\n".join(pip_requirements))


# COMMAND ----------

from typing import Any

from mlflow.models import ModelConfig

model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

catalog_config: dict[str, Any] = config.get("catalog")
catalog_name: str = catalog_config.get("catalog_name")
database_name: str = catalog_config.get("database_name")
volume_name: str = catalog_config.get("volume_name")

data_config: dict[str, Any] = config.get("data")
raw_email_path: str = data_config.get("raw_email_path")
source_table_name: str = data_config.get("source_table_name")
primary_key: str = data_config.get("primary_key")


assert catalog_name is not None
assert database_name is not None
assert volume_name is not None
assert raw_email_path is not None
assert source_table_name is not None
assert primary_key is not None


# COMMAND ----------

from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
  CatalogInfo,
  SchemaInfo,
  VolumeInfo,
  VolumeType,
  SecurableType,
  PermissionsChange,
  Privilege
)


def _volume_as_path(self: VolumeInfo) -> Path:
  return Path(f"/Volumes/{self.catalog_name}/{self.schema_name}/{self.name}")

# monkey patch
VolumeInfo.as_path = _volume_as_path


w: WorkspaceClient = WorkspaceClient()

catalog: CatalogInfo
try:
  catalog = w.catalogs.get(catalog_name)
except Exception as e:
  catalog = w.catalogs.create(catalog_name)

schema: SchemaInfo
try:
  schema = w.schemas.get(f"{catalog.full_name}.{database_name}")
except Exception as e:
  schema = w.schemas.create(database_name, catalog.full_name)

volume: VolumeInfo
try:
  volume = w.volumes.read(f"{catalog.full_name}.{database_name}.{volume_name}")
except Exception as e:
  volume = w.volumes.create(catalog.full_name, schema.name, volume_name, VolumeType.MANAGED)

spark.sql(f"USE {schema.full_name}")

# COMMAND ----------

import re

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T

from delta.tables import DeltaTable, IdentityGenerator


def process_email(df: DataFrame) -> DataFrame:
  processed_email_df: DataFrame = df.select(df.sender, df.subject, df.body, df.recipients)
  return processed_email_df

raw_email_df: DataFrame = spark.read.json(raw_email_path)

email_df: DataFrame = process_email(raw_email_df)

(
  DeltaTable.createOrReplace(spark)
    .tableName(source_table_name)
    .property("delta.enableChangeDataFeed", "true")
    .addColumn(primary_key, dataType=T.LongType(), nullable=False, generatedAlwaysAs=IdentityGenerator())
    .addColumns(email_df.schema)
    .execute()
)

spark.sql(f"ALTER TABLE {source_table_name} ADD CONSTRAINT {primary_key}_pk PRIMARY KEY ({primary_key})")

email_df.write.mode("append").saveAsTable(source_table_name)

display(spark.table(source_table_name))
