"""Reddit client for fetching Polymarket-related posts and comments."""
from datetime import datetime, timezone

import pandas as pd
import praw
from loguru import logger

from src.config.settings import config


class RedditClient:
    """Fetches posts and comments from prediction-market-related subreddits."""

    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=config.reddit.client_id,
            client_secret=config.reddit.client_secret,
            user_agent=config.reddit.user_agent,
        )
        self.subreddits = config.reddit.subreddits

    def fetch_posts(
        self,
        query: str,
        subreddit_name: str = None,
        limit: int = 500,
        sort: str = "relevance",
        time_filter: str = "year",
    ) -> pd.DataFrame:
        """
        Search Reddit for posts matching a market question.

        Args:
            query: Search query (usually derived from market question)
            subreddit_name: Specific subreddit or None for all configured
            limit: Max posts to fetch
            sort: relevance, hot, top, new, comments
            time_filter: hour, day, week, month, year, all

        Returns DataFrame with: id, title, selftext, score, num_comments,
                                 created_utc, subreddit, author, url
        """
        subreddits = [subreddit_name] if subreddit_name else self.subreddits
        all_posts = []

        for sub_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(sub_name)
                results = subreddit.search(
                    query, sort=sort, time_filter=time_filter, limit=limit
                )

                for post in results:
                    all_posts.append({
                        "id": post.id,
                        "title": post.title,
                        "selftext": post.selftext or "",
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(
                            post.created_utc, tz=timezone.utc
                        ),
                        "subreddit": sub_name,
                        "author": str(post.author) if post.author else "[deleted]",
                        "url": post.url,
                        "upvote_ratio": post.upvote_ratio,
                    })

            except Exception as e:
                logger.error(f"Error fetching from r/{sub_name}: {e}")
                continue

        df = pd.DataFrame(all_posts)
        logger.info(f"Fetched {len(df)} Reddit posts for query: '{query[:50]}...'")
        return df

    def fetch_comments(self, post_id: str, limit: int = 200) -> pd.DataFrame:
        """Fetch comments for a specific post."""
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)

        comments = []
        for comment in submission.comments.list()[:limit]:
            comments.append({
                "id": comment.id,
                "post_id": post_id,
                "body": comment.body,
                "score": comment.score,
                "created_utc": datetime.fromtimestamp(
                    comment.created_utc, tz=timezone.utc
                ),
                "author": str(comment.author) if comment.author else "[deleted]",
            })

        return pd.DataFrame(comments)
