val conf = spark.sparkContext.getConf

// data scale in GB
val scaleFactor = conf.getInt("spark.driver.scaleFactor", 1)
// support parquet or orc
val format = conf.get("spark.driver.format", "parquet")
// create partitioned table
val partitionTables = conf.getBoolean("spark.driver.partitioned", true)
// how many times to run the whole set of queries.
val iterations = conf.getInt("spark.driver.iterations", 1)
// support s3a://s3_bucket, gs://gs_bucket
// wasbs://container@storage_account.blob.core.windows.net
// abfs://container@storage_account.dfs.core.windows.net
val fsdir = conf.get("spark.driver.fsdir", "")
// If the tables in database are not fully created in the previous run, you need to force to drop and recreate the database and tables.
val recreateDatabase = conf.getBoolean("spark.driver.recreateDatabase", false)

val query_filter = Seq()        // Seq() == all queries
// val query_filter = Seq("q1-v2.4", "q2-v2.4") // run subset of queries
val randomizeQueries = false    // run queries in a random order. Recommended for parallel runs.

if (fsdir == "") {
    println("File system dir must be specified with --conf spark.driver.fsdir")
    sys.exit(0)
}

// detailed results will be written as JSON to this location.
var resultLocation = s"${fsdir}/shared/data/results/tpcds_${format}/${scaleFactor}"
val data_path = s"${fsdir}/shared/data/tpcds/tpcds_${format}/${scaleFactor}"
var databaseName = s"tpcds_${format}_scale_${scaleFactor}_db"
val use_arrow = conf.getBoolean("spark.driver.useArrow", false) // when you want to use gazella_plugin to run TPC-DS, you need to set it true.

if (use_arrow){
    resultLocation = s"${fsdir}/shared/data/results/tpcds_arrow/${scaleFactor}/"
    databaseName = s"tpcds_arrow_scale_${scaleFactor}_db"
    val tables = Seq("call_center", "catalog_page", "catalog_returns", "catalog_sales", "customer", "customer_address", "customer_demographics", "date_dim", "household_demographics", "income_band", "inventory", "item", "promotion", "reason", "ship_mode", "store", "store_returns", "store_sales", "time_dim", "warehouse", "web_page", "web_returns", "web_sales", "web_site")
    if (spark.catalog.databaseExists(s"$databaseName")) {
        if (!recreateDatabase) {
            println(s"Using existing $databaseName")
        } else {
            println(s"$databaseName exists, now drop and recreate it...")
            sql(s"drop database if exists $databaseName cascade")
            sql(s"create database if not exists $databaseName").show
        }
    } else {
        println(s"$databaseName doesn't exist. Creating...")
        sql(s"create database if not exists $databaseName").show
    }
    sql(s"use $databaseName").show
    for (table <- tables) {
        if (spark.catalog.tableExists(s"$table")){
            println(s"$table exists.")
        }else{
            spark.catalog.createTable(s"$table", s"$data_path/$table", "arrow")
        }
    }
    if (partitionTables) {
        for (table <- tables) {
            try{
                sql(s"ALTER TABLE $table RECOVER PARTITIONS").show
            }catch{
                case e: Exception => println(e)
            }
        }
    }
} else {
    // Check whether the database is created, we create external tables if not
    val databaseExists = spark.catalog.databaseExists(s"$databaseName")
    if (databaseExists && !recreateDatabase) {
        println(s"Using existing $databaseName")
    } else {
        if (databaseExists) {
            println(s"$databaseName exists, now drop and recreate it...")
            sql(s"drop database if exists $databaseName cascade")
        } else {
            println(s"$databaseName doesn't exist. Creating...")
        }

        import com.databricks.spark.sql.perf.tpcds.TPCDSTables

        val tables = new TPCDSTables(spark.sqlContext, "", s"${scaleFactor}", false)
        tables.createExternalTables(data_path, format, databaseName, overwrite = true, discoverPartitions = partitionTables)
    }
}

val timeout = 60 // timeout in hours

// COMMAND ----------

// Spark configuration
spark.conf.set("spark.sql.broadcastTimeout", "10000") // good idea for Q14, Q88.

// ... + any other configuration tuning

// COMMAND ----------
sql(s"use $databaseName")
import com.databricks.spark.sql.perf.tpcds.TPCDS
val tpcds = new TPCDS (sqlContext = spark.sqlContext)
def queries = {
  val filtered_queries = query_filter match {
    case Seq() => tpcds.tpcds2_4Queries
    case _ => tpcds.tpcds2_4Queries.filter(q => query_filter.contains(q.name))
  }
  if (randomizeQueries) scala.util.Random.shuffle(filtered_queries) else filtered_queries
}
val experiment = tpcds.runExperiment(
  queries,
  iterations = iterations,
  resultLocation = resultLocation,
  tags = Map("runtype" -> "benchmark", "database" -> databaseName, "scale_factor" -> s"${scaleFactor}"))

println(experiment.toString)
experiment.waitForFinish(timeout*60*60)

// Process general performance results
val resultPath = experiment.resultPath
val resultDF = spark.read.json(resultPath)
val result = resultDF.withColumn("result", explode(col("results")))
  .withColumn("Name", substring(col("result.name"), 1, 100))
  .withColumn("Runtime", round(((col("result.parsingTime") + col("result.analysisTime") + col("result.optimizationTime") + col("result.planningTime") + col("result.executionTime")) / 1000.0), 2))
  .select("Iteration", "Name", "Runtime")

// Present all iterations performance results to columns
import org.apache.spark.sql.DataFrame
var fullResult: DataFrame = result.select(col("Name").as("Query")).filter("Iteration = 1")
for( r <- 1 to iterations) {
  val roundResult = result.filter(f"Iteration = $r").withColumn(f"Round$r", col("Runtime"))
  fullResult = fullResult.join(roundResult, fullResult("Query") === roundResult("Name")).drop("Iteration", "Name", "Runtime")
}

// Calculate and present the query's maximum, minimum and average runtime of each round.
val calResult = result.groupBy("Name").agg(max("Runtime").as("Max"), min("Runtime").as("Min"), round(avg("Runtime"),2).as("Average"))
fullResult = fullResult.join(calResult, fullResult("Query") === calResult("Name")).drop("Name")

val columns = fullResult.columns.dropWhile(_ == "Query").map(col)
val totalResult = fullResult.union(fullResult.select(lit("Total").as("Query") +: columns.map(sum):_*))

val roundCols = totalResult.columns.filter(!_.startsWith("Query"))
val finalResult = totalResult.select(col("Query") +: roundCols.map(c => round(col(c), 2).as(c)): _*)

finalResult.show(200)

// Save the performance summary dataframe to a CSV file with a specified file name
val finalResultPath = s"${experiment.resultPath}/summary"
finalResult.repartition(1).write.option("header", "true").mode("overwrite").csv(finalResultPath)

import org.apache.hadoop.fs.{FileSystem, Path}
import java.net.URI

val fs = FileSystem.get(URI.create(finalResultPath), sc.hadoopConfiguration)
val file = fs.globStatus(new Path(s"$finalResultPath/*.csv"))(0).getPath().getName()
val srcPath=new Path(s"$finalResultPath/$file")
val destPath= new Path(s"$finalResultPath/summary.csv")
fs.rename(srcPath, destPath)

println(s"Performance summary is saved to ${destPath}")

sys.exit(0)
