"""
Mock data stream for Trend Detector (demo mode).
"""

import random
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

from shared.lib.config import get_settings
from shared.lib.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class MockStream(LoggerMixin):
    """Mock data stream generator for demo purposes."""

    def __init__(self, sources: list[str] = None):
        self.sources = sources or ["reddit", "twitter"]
        self.settings = get_settings()

        # Predefined topics and keywords
        self.topics = {
            "technology": [
                "AI",
                "machine learning",
                "artificial intelligence",
                "deep learning",
                "neural networks",
                "computer vision",
                "NLP",
                "robotics",
                "automation",
                "blockchain",
                "cryptocurrency",
                "bitcoin",
                "ethereum",
                "DeFi",
                "quantum computing",
                "IoT",
                "5G",
                "cloud computing",
                "cybersecurity",
            ],
            "health": [
                "vaccine",
                "COVID",
                "pandemic",
                "healthcare",
                "medicine",
                "research",
                "clinical trial",
                "treatment",
                "therapy",
                "diagnosis",
                "prevention",
                "mental health",
                "wellness",
                "fitness",
                "nutrition",
                "diet",
                "exercise",
                "yoga",
                "meditation",
                "stress",
                "anxiety",
            ],
            "climate": [
                "climate change",
                "global warming",
                "carbon emissions",
                "renewable energy",
                "solar power",
                "wind energy",
                "sustainability",
                "green energy",
                "electric vehicles",
                "Tesla",
                "environment",
                "pollution",
                "carbon footprint",
                "recycling",
                "conservation",
                "biodiversity",
            ],
            "politics": [
                "election",
                "vote",
                "democracy",
                "government",
                "policy",
                "legislation",
                "congress",
                "senate",
                "president",
                "prime minister",
                "parliament",
                "campaign",
                "candidate",
                "political party",
                "reform",
                "protest",
            ],
            "entertainment": [
                "movie",
                "film",
                "cinema",
                "Netflix",
                "streaming",
                "TV show",
                "celebrity",
                "actor",
                "actress",
                "director",
                "uber",
                "music",
                "song",
                "album",
                "concert",
                "festival",
                "game",
                "gaming",
                "esports",
                "sports",
                "football",
                "basketball",
                "soccer",
            ],
            "finance": [
                "stock market",
                "trading",
                "investment",
                "portfolio",
                "crypto",
                "trading",
                "forex",
                "economy",
                "inflation",
                "recession",
                "GDP",
                "unemployment",
                "federal reserve",
                "interest rates",
                "real estate",
                "mortgage",
                "loan",
                "credit",
                "banking",
            ],
        }

        # Sample posts for each topic
        self.sample_posts = {
            "technology": [
                "Just discovered this amazing new AI tool that can generate code automatically! The future is here.",
                "Machine learning is revolutionizing healthcare. These algorithms can detect diseases earlier than ever.",
                "The latest breakthrough in quantum computing could change everything we know about encryption.",
                "OpenAI's GPT-4 is incredible. It's like having a conversation with a superintelligent being.",
                "Robotics and automation are transforming manufacturing. The efficiency gains are remarkable.",
            ],
            "health": [
                "New study shows promising results for Alzheimer's treatment. Hope for millions of families.",
                "Mental health awareness is so important. Let's break the stigma and support each other.",
                "Exercise has incredible benefits for both physical and mental health. Even 30 minutes makes a difference.",
                "The importance of sleep for cognitive function cannot be overstated. Quality over quantity.",
                "Meditation and mindfulness practices are backed by solid scientific research.",
            ],
            "climate": [
                "Renewable energy adoption is accelerating faster than expected. Solar and wind are becoming mainstream.",
                "Climate change is the defining challenge of our time. We need immediate action on all fronts.",
                "Electric vehicles are finally becoming affordable and practical for everyday use.",
                "The transition to a green economy is creating millions of new jobs worldwide.",
                "Individual actions matter, but systemic change is what we really need.",
            ],
            "politics": [
                "Democracy requires active participation from all citizens. Your vote truly matters.",
                "Policy decisions today will shape the world for decades to come. We need long-term thinking.",
                "Transparency and accountability in government are essential for a healthy democracy.",
                "The importance of fact-checking and critical thinking in political discourse cannot be overstated.",
                "Civic engagement goes beyond voting. Stay informed and participate in your community.",
            ],
            "entertainment": [
                "The new season of this show is absolutely incredible. The character development is outstanding.",
                "Music has the power to bring people together across all boundaries and differences.",
                "Gaming has evolved into a legitimate art form with incredible storytelling capabilities.",
                "Streaming services are changing how we consume entertainment. The variety is amazing.",
                "Sports bring communities together and teach valuable life lessons about teamwork and perseverance.",
            ],
            "finance": [
                "Diversification is key to building a resilient investment portfolio for the long term.",
                "Understanding compound interest is one of the most important financial concepts to master.",
                "The cryptocurrency market is highly volatile. Only invest what you can afford to lose.",
                "Real estate remains one of the most stable long-term investment options available.",
                "Financial literacy should be taught in schools. It's essential for economic well-being.",
            ],
        }

        # Trending keywords that appear more frequently
        self.trending_keywords = [
            "AI",
            "climate",
            "crypto",
            "health",
            "tech",
            "innovation",
            "sustainability",
            "mental health",
            "renewable energy",
            "automation",
        ]

        self.logger.info("Mock stream initialized with demo data")

    def generate_post(self, topic: str, source: str) -> dict[str, Any]:
        """Generate a single mock post."""

        # Get topic keywords and sample posts
        keywords = self.topics.get(topic, [])
        sample_posts = self.sample_posts.get(topic, [])

        # Generate post text
        if sample_posts and random.random() < 0.7:
            # Use sample post 70% of the time
            text = random.choice(sample_posts)
        else:
            # Generate random text 30% of the time
            text = self._generate_random_text(keywords)

        # Add trending keywords occasionally
        if random.random() < 0.3:
            trending_word = random.choice(self.trending_keywords)
            text = f"{text} #{trending_word}"

        # Generate metadata
        now = datetime.now()
        timestamp = now - timedelta(
            minutes=random.randint(0, 1440),  # Last 24 hours
            seconds=random.randint(0, 59),
        )

        post = {
            "text": text,
            "source": source,
            "timestamp": timestamp.isoformat(),
            "url": f"https://{source}.com/post/{random.randint(100000, 999999)}",
            "author": f"user_{random.randint(1000, 9999)}",
            "score": random.randint(1, 1000),
            "topic": topic,
            "keywords": random.sample(keywords, min(3, len(keywords))),
            "sentiment": random.choice(["positive", "negative", "neutral"]),
            "language": "en",
        }

        return post

    def _generate_random_text(self, keywords: list[str]) -> str:
        """Generate random text using keywords."""

        # Simple templates
        templates = [
            "Just learned about {keyword}. This is fascinating!",
            "Anyone else interested in {keyword}? Thoughts?",
            "The future of {keyword} looks promising.",
            "Discussion about {keyword} - what do you think?",
            "Exciting developments in {keyword} technology.",
            "{keyword} is changing everything we know.",
            "Hot take: {keyword} will be huge this year.",
            "Can't stop thinking about {keyword} possibilities.",
            "The {keyword} revolution is here.",
            "Why {keyword} matters more than you think.",
        ]

        keyword = random.choice(keywords) if keywords else "technology"
        template = random.choice(templates)

        return template.format(keyword=keyword)

    def stream_posts(
        self, duration_minutes: int = 60, posts_per_minute: int = 5
    ) -> Iterator[dict[str, Any]]:
        """Stream mock posts for a specified duration."""

        self.logger.info(
            f"Starting mock stream for {duration_minutes} minutes at {posts_per_minute} posts/minute"
        )

        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        while datetime.now() < end_time:
            # Generate posts for this minute
            posts_this_minute = random.randint(
                max(1, posts_per_minute - 2), posts_per_minute + 2
            )

            for _ in range(posts_this_minute):
                # Select random topic and source
                topic = random.choice(list(self.topics.keys()))
                source = random.choice(self.sources)

                # Generate post
                post = self.generate_post(topic, source)

                yield post

            # Wait for next minute (simulated)
            import time

            time.sleep(1)  # In demo mode, we don't wait the full minute

    def get_trending_topics(self, window_hours: int = 24) -> list[dict[str, Any]]:
        """Get trending topics based on mock data."""

        trending = []

        for topic, keywords in self.topics.items():
            # Simulate trending score
            base_score = random.uniform(0.1, 0.9)

            # Boost trending keywords
            if any(kw in self.trending_keywords for kw in keywords):
                base_score += 0.2

            # Add some volatility
            volatility = random.uniform(-0.1, 0.1)
            final_score = max(0, min(1, base_score + volatility))

            trending.append(
                {
                    "topic": topic,
                    "keywords": keywords[:5],  # Top 5 keywords
                    "trending_score": final_score,
                    "volume": random.randint(50, 500),
                    "growth_rate": random.uniform(-0.2, 0.5),
                    "representative_posts": random.sample(
                        self.sample_posts.get(topic, []),
                        min(3, len(self.sample_posts.get(topic, []))),
                    ),
                }
            )

        # Sort by trending score
        trending.sort(key=lambda x: x["trending_score"], reverse=True)

        return trending

    def get_recent_posts(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent mock posts."""

        posts = []

        for _ in range(limit):
            topic = random.choice(list(self.topics.keys()))
            source = random.choice(self.sources)
            post = self.generate_post(topic, source)
            posts.append(post)

        # Sort by timestamp (most recent first)
        posts.sort(key=lambda x: x["timestamp"], reverse=True)

        return posts

    def simulate_burst(
        self, topic: str, duration_minutes: int = 30
    ) -> Iterator[dict[str, Any]]:
        """Simulate a topic burst (sudden increase in activity)."""

        self.logger.info(f"Simulating burst for topic: {topic}")

        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        while datetime.now() < end_time:
            # Generate more posts during burst
            posts_this_interval = random.randint(10, 20)

            for _ in range(posts_this_interval):
                source = random.choice(self.sources)
                post = self.generate_post(topic, source)
                yield post

            # Shorter intervals during burst
            import time

            time.sleep(0.1)

    def get_source_stats(self) -> dict[str, Any]:
        """Get statistics about mock sources."""

        stats = {}

        for source in self.sources:
            stats[source] = {
                "total_posts": random.randint(1000, 10000),
                "active_users": random.randint(100, 1000),
                "avg_score": random.uniform(5, 50),
                "top_topics": random.sample(list(self.topics.keys()), 3),
            }

        return stats


def create_mock_stream(sources: list[str] = None) -> MockStream:
    """Create a mock stream instance."""
    return MockStream(sources)


def generate_sample_data(num_posts: int = 1000) -> list[dict[str, Any]]:
    """Generate sample data for testing."""

    stream = MockStream()
    posts = []

    for _ in range(num_posts):
        topic = random.choice(list(stream.topics.keys()))
        source = random.choice(stream.sources)
        post = stream.generate_post(topic, source)
        posts.append(post)

    return posts


if __name__ == "__main__":
    # Demo the mock stream
    stream = MockStream()

    print("Generating sample posts...")
    for i, post in enumerate(
        stream.stream_posts(duration_minutes=1, posts_per_minute=2)
    ):
        print(f"Post {i+1}: {post['text'][:50]}...")
        if i >= 10:  # Limit for demo
            break

    print("\nTrending topics:")
    trending = stream.get_trending_topics()
    for topic in trending[:5]:
        print(f"- {topic['topic']}: {topic['trending_score']:.3f}")
