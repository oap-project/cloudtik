# Spark Optimizations
CloudTik Spark can run on both the up-stream Spark and a CloudTik optimized Spark.  
CloudTik optimized Spark implemented quite a few important optimizations upon
the up-stream Spark and thus provides better performance.

- [Runtime Filter Optimization](#runtime-filter-optimization)
- [Top N Optimization](#top-n-optimization)
- [Size-based Join Reorder Optimization](#size-based-join-reorder-optimization)
- [Distinct Before Intersect Optimization](#distinct-before-intersect-optimization)
- [Flatten Scalar Subquery Optimization](#flatten-scalar-subquery-optimization)
- [Flatten Single Row Aggregate Optimization](#flatten-single-row-aggregate-optimization)

## Runtime Filter Optimization
Row-level runtime filters can improve the performance of some joins by pre-filtering one side (Filter Application Side)
of a join using a Bloom filter or semi-join filters generated from the values from the other side (Filter Creation Side) of the join. 

## Top N Optimization
For the rank functions (row_number|rank|dense_rank),
the rank of a key computed on partial dataset is always <= its final rank computed on the whole dataset.
It’s safe to discard rows with partial rank > k.  Select local top-k records within each partition,
and then compute the global top-k. This can help reduce the shuffle amount.
We introduce a new node RankLimit to filter out unnecessary rows based on rank computed on partial dataset.
We can enable this feature by setting spark.sql.rankLimit.enabled to true.

## Size-based Join Reorder Optimization
The default behavior in Spark is to join tables from left to right, as listed in the query.
We can improve query performance by reordering joins involving tables with filters.
You can enable this feature by setting the Spark configuration parameter spark.sql.optimizer.sizeBasedJoinReorder.enabled to true.

## Distinct Before Intersect Optimization
This optimization optimizes joins when using INTERSECT.
Queries using INTERSECT are automatically converted to use a left-semi join.
When this optimization is enabled, the query optimizer will try to estimate whether pushing the DISTINCT operator
to the children of INTERSECT has benefit according to the duplication of data in the left table and the right table.
You can enable it by setting the Spark property spark.sql.optimizer.distinctBeforeIntersect.enabled.

## Flatten Scalar Subquery Optimization

## Flatten Single Row Aggregate Optimization

