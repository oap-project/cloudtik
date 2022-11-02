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
- [Remove Duplicate Joins InSubquery](#remove-duplicate-joins-InSubquery)

## Runtime Filter Optimization
Row-level runtime filters can improve the performance of some joins by pre-filtering one side (Filter Application Side)
of a join using a Bloom filter or semi-join filters generated from the values from the other side (Filter Creation Side) of the join. 

## Top N Optimization
For the rank functions (row_number|rank|dense_rank),
the rank of a key computed on partial dataset is always <= its final rank computed on the whole dataset.
It’s safe to discard rows with partial rank > k.  Select local top-k records within each partition,
and then compute the global top-k. This can help reduce the shuffle amount.
We introduce a new node RankLimit to filter out unnecessary rows based on rank computed on partial dataset.
We can enable this feature by setting ```spark.sql.rankLimit.enabled``` to ```true```.

## Size-based Join Reorder Optimization
The default behavior in Spark is to join tables from left to right, as listed in the query.
We can improve query performance by reordering joins involving tables with filters.
You can enable this feature by setting the Spark configuration parameter ```spark.sql.optimizer.sizeBasedJoinReorder.enabled``` to ```true```.

## Distinct Before Intersect Optimization
This optimization optimizes joins when using INTERSECT.
Queries using INTERSECT are automatically converted to use a left-semi join.
When this optimization is enabled, the query optimizer will try to estimate whether pushing the DISTINCT operator
to the children of INTERSECT has benefit according to the duplication of data in the left table and the right table.
You can enable it by setting the Spark property ```spark.sql.optimizer.distinctBeforeIntersect.enabled``` to ```true```.

## Flatten Scalar Subquery Optimization
We add a new optimizer rule MergeScalarSubqueries to merge multiple non-correlated ScalarSubquerys to compute multiple scalar values once.
The query optimizer flattens aggregate scalar subqueries that use the same relation if possible. 
The scalar subqueries are flattened by pushing any predicates present in the subquery into the aggregate functions and then performing one aggregation,
 with all the aggregate functions, per relation.
 
## Flatten Single Row Aggregate Optimization
This optimization is similar with [Flatten Scalar Subquery Optimization](#flatten-scalar-subquery-optimization). For cross join, the children may be both aggregate with single row.
The query optimizer flattens aggregate nodes of cross join that return one row and use the same relation if possible. 
If the children of cross join can be merged, we will replace the cross join by merged node.

## Remove Duplicate Joins InSubquery
This optimization will try to find useless joins of all InSubqueries and remove them.  You can enable this feature by setting 
the Spark property ```spark.sql.optimizer.removeInSubqueryDuplicateJoins.enabled``` to ```true```. The following is an example query
that can benefit from this optimization.
```
select
   count(distinct ws_order_number) as `order count`
from
   web_sales ws1
where
   ws1.ws_order_number in (select ws_order_number
                            from ws_wh)
and ws1.ws_order_number in (select wr_order_number
                            from web_returns,ws_wh
                            where wr_order_number = ws_wh.ws_order_number)
order by count(distinct ws_order_number)
limit 100
```
When this feature is enabled,  duplicate joins of the InSubqueries which contain the same values will be found. 
Removing these joins can reduce shuffle data.
```
select
   count(distinct ws_order_number) as `order count`
from
   web_sales ws1
where
   ws1.ws_order_number in (select ws_order_number
                            from ws_wh)
and ws1.ws_order_number in (select wr_order_number
                            from web_returns)
order by count(distinct ws_order_number)
limit 100
```