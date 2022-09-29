import java.text.SimpleDateFormat;
import java.util.Date
import java.util.concurrent.Executors
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import com.databricks.spark.sql.perf.tpcds.TPCDS
import com.databricks.spark.sql.perf.Benchmark.ExperimentStatus
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, lit, substring}

val conf = spark.sparkContext.getConf

// how many streams you want to start
val streamNumber = conf.getInt("spark.driver.streamNumber", 2)
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
val randomizeQueries = true    // run queries in a random order. Recommended for parallel runs.

if (fsdir == "") {
    println("File system dir must be specified with --conf spark.driver.fsdir")
    sys.exit(0)
}


val current_time = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date)
// detailed results will be written as JSON to this location.
var resultLocation = s"${fsdir}/shared/data/results/tpcds_${format}/${scaleFactor}/${current_time}"
val data_path = s"${fsdir}/shared/data/tpcds/tpcds_${format}/${scaleFactor}"
var databaseName = s"tpcds_${format}_scale_${scaleFactor}_db"
val use_arrow = conf.getBoolean("spark.driver.useArrow", false) // when you want to use gazella_plugin to run TPC-DS, you need to set it true.

if (use_arrow){
    resultLocation = s"${fsdir}/shared/data/results/tpcds_arrow/${scaleFactor}/${current_time}"
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
val tpcds = new TPCDS (sqlContext = spark.sqlContext)
def queries(stream: Int) = {
  val filtered_queries = query_filter match {
    case Seq() => tpcds.tpcds2_4Queries
    case _ => tpcds.tpcds2_4Queries.filter(q => query_filter.contains(q.name))
  }
  if (randomizeQueries && stream != 0) scala.util.Random.shuffle(filtered_queries) else filtered_queries
}

class ThreadStream(experiment:ExperimentStatus, i:Int) extends Thread{
    override def run(){
        println("stream_" + i + " has started...")
        println(experiment.toString)
        experiment.waitForFinish(timeout*60*60)
        println("stream_" + i + " has finished.")
    }
}

val threadPool:ExecutorService=Executors.newFixedThreadPool(streamNumber)
val experiments:Array[ExperimentStatus] = new Array[ExperimentStatus](streamNumber)

try {
    for(i <- 0 to (streamNumber - 1)){
        experiments(i) = tpcds.runExperiment(
            queries(i),
            iterations = iterations,
            resultLocation = resultLocation,
            tags = Map("runtype" -> "benchmark", "database" -> databaseName, "scale_factor" -> s"${scaleFactor}")
        )
        threadPool.execute(new ThreadStream(experiments(i), i))
    }
}finally{
    threadPool.shutdown()
    threadPool.awaitTermination(Long.MaxValue, TimeUnit.NANOSECONDS)
}

val summary_dfs = new Array[DataFrame](streamNumber)
for(i <- 0 to (streamNumber - 1)){
   summary_dfs(i) = experiments(i).getCurrentResults.withColumn("Name", substring(col("name"), 2, 100)).withColumn("Runtime", (col("parsingTime") + col("analysisTime") + col("optimizationTime") + col("planningTime") + col("executionTime")) / 1000.0).select('Name, 'Runtime).agg(sum("Runtime")).withColumn("stream", lit("stream_" + i)).select("stream", "sum(Runtime)")
}
var summary_df = summary_dfs(0)
for (i <- 0 to (streamNumber - 1)){
    if (i != 0) {
        summary_df = summary_df.union(summary_dfs(i))
    }
}
summary_df = summary_df.union(summary_df.agg(max("sum(Runtime)")).withColumnRenamed("max(sum(Runtime))","sum(Runtime)").withColumn("stream", lit("max_stream")).select("stream", "sum(Runtime)"))
summary_df.show()

// Save the performance summary dataframe to a CSV file with a specified file name
val finalResultPath = s"${resultLocation}/summary"
summary_df.repartition(1).write.option("header", "true").mode("overwrite").csv(finalResultPath)

import org.apache.hadoop.fs.{FileSystem, Path}
import java.net.URI

val fs = FileSystem.get(URI.create(finalResultPath), sc.hadoopConfiguration)
val file = fs.globStatus(new Path(s"$finalResultPath/*.csv"))(0).getPath().getName()
val srcPath=new Path(s"$finalResultPath/$file")
val destPath= new Path(s"$finalResultPath/summary.csv")
fs.rename(srcPath, destPath)

println(s"Performance summary is saved to ${destPath}")

sys.exit(0)
