-- Instamart Capstone — Part A SQL analysis
-- Run against data/instamart.db. Completed orders count as revenue/spend.

-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS qualifying_spend
FROM orders
WHERE LOWER(status) = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend DESC;

-- Q2: Three-table join — revenue by city and product category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE LOWER(o.status) = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: Scalar subquery — products above average catalogue price
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN — customers with no completed order
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
  ON o.customer_id = c.customer_id AND LOWER(o.status) = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: CTEs — customers whose completed spend exceeds the customer average
WITH customer_spend AS (
    SELECT c.customer_id, c.city, COALESCE(SUM(o.order_amount), 0) AS total_spend
    FROM customers c
    LEFT JOIN orders o
      ON o.customer_id = c.customer_id AND LOWER(o.status) = 'completed'
    GROUP BY c.customer_id, c.city
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_spend
)
SELECT cs.customer_id, cs.city, cs.total_spend, a.avg_spend
FROM customer_spend cs CROSS JOIN average_spend a
WHERE cs.total_spend > a.avg_spend
ORDER BY cs.total_spend DESC;

-- Q6: Ranking window functions — customer spend rank within each city
WITH customer_spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE LOWER(o.status) = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS row_number
FROM customer_spend
ORDER BY city, spend_rank, customer_id;

-- Q7: Daily completed revenue and cumulative running total
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE LOWER(status) = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
