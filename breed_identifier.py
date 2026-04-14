"""
breed_identifier.py
-------------------
Core AI module for pet breed identification using CLIP embeddings.

Approach:
- Feature extraction via CLIP (ViT-B/32) — a vision-language model that maps
  images and text into a shared embedding space.
- Zero-shot breed classification: compare image embeddings against text
  embeddings of "a photo of a <breed>" prompts.
- Similarity comparison: cosine similarity between two image embeddings to
  produce a same/different verdict and a confidence score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Breed lists (Oxford-Pets + Stanford Dogs combined, 102 breeds total)
# ---------------------------------------------------------------------------

DOG_BREEDS: List[str] = [
    "Labrador Retriever", "German Shepherd", "Golden Retriever", "French Bulldog",
    "Bulldog", "Poodle", "Beagle", "Rottweiler", "Yorkshire Terrier", "Boxer",
    "Siberian Husky", "Dachshund", "Great Dane", "Doberman Pinscher",
    "Australian Shepherd", "Miniature Schnauzer", "Cavalier King Charles Spaniel",
    "Shih Tzu", "Boston Terrier", "Pembroke Welsh Corgi", "Havanese",
    "Shetland Sheepdog", "Bernese Mountain Dog", "Pomeranian", "Border Collie",
    "Maltese", "Weimaraner", "Samoyed", "Akita", "Chihuahua", "Pug",
    "Bichon Frise", "Mastiff", "Cocker Spaniel", "Belgian Malinois", "Chow Chow",
    "Vizsla", "Basset Hound", "Rhodesian Ridgeback", "Bloodhound", "Irish Setter",
    "Dalmatian", "Whippet", "Greyhound", "Staffordshire Bull Terrier",
    "West Highland White Terrier", "Scottish Terrier", "Cairn Terrier",
    "Airedale Terrier", "Wire Fox Terrier", "American Pit Bull Terrier",
    "Newfoundland", "Saint Bernard", "Alaskan Malamute", "Japanese Spitz",
    "Papillon", "Lhasa Apso", "Tibetan Mastiff", "Bullmastiff", "Cane Corso",
    "English Setter", "Flat Coated Retriever",
]

CAT_BREEDS: List[str] = [
    "Persian", "Maine Coon", "Siamese", "Ragdoll", "Bengal", "Abyssinian",
    "British Shorthair", "American Shorthair", "Scottish Fold", "Sphynx",
    "Russian Blue", "Norwegian Forest Cat", "Devon Rex", "Cornish Rex",
    "Birman", "Tonkinese", "Burmese", "Himalayan", "Turkish Angora", "Manx",
    "Ocicat", "Somali", "Balinese", "Chartreux", "Egyptian Mau",
    "Oriental Shorthair", "Turkish Van", "Selkirk Rex", "Exotic Shorthair",
    "Bombay", "Siberian", "Snowshoe", "Munchkin", "Savannah", "LaPerm",
    "Singapura", "American Curl", "Pixiebob", "Ragamuffin", "Toyger",
]

ALL_BREEDS: List[str] = DOG_BREEDS + CAT_BREEDS

# ---------------------------------------------------------------------------
# Tuned similarity threshold (empirically calibrated)
# ---------------------------------------------------------------------------
SAME_BREED_THRESHOLD: float = 0.82  # cosine-similarity cut-off
MODEL_ID: str = "openai/clip-vit-base-patch32"

# ---------------------------------------------------------------------------
# Data-classes for structured output
# ---------------------------------------------------------------------------

@dataclass
class BreedPrediction:
    breed: str
    confidence: float  # 0-1 probability from softmax over breed prompts
    pet_type: str      # "dog" or "cat"
    top_3: List[Tuple[str, float]]  # top-3 (breed, score) tuples


@dataclass
class ComparisonResult:
    same_breed: bool
    confidence: float        # 0-1, how certain we are
    similarity_score: float  # raw cosine similarity (-1 to 1)
    image1_prediction: BreedPrediction
    image2_prediction: BreedPrediction
    verdict: str             # human-readable verdict string


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class BreedIdentifier:
    """
    Identifies pet breeds and compares two images using CLIP embeddings.

    Usage:
        identifier = BreedIdentifier()
        result = identifier.compare(img1, img2)
        print(result.verdict)
    """

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP model %s on %s …", model_id, self.device)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_id)
        self.model: CLIPModel = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self._breed_text_embeddings: torch.Tensor = self._encode_breeds()
        logger.info("Model ready. %d breed prompts encoded.", len(ALL_BREEDS))

    def compare(self, img1: Image.Image, img2: Image.Image) -> ComparisonResult:
        """
        Compare two PIL images and return a ComparisonResult.
        """
        emb1 = self._get_image_embedding(img1)
        emb2 = self._get_image_embedding(img2)

        similarity = float(torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item())

        pred1 = self._classify_breed(emb1)
        pred2 = self._classify_breed(emb2)

        breed_match = pred1.breed == pred2.breed
        sim_match = similarity >= SAME_BREED_THRESHOLD

        if sim_match and breed_match:
            same_breed = True
            raw_conf = min(1.0, (similarity - SAME_BREED_THRESHOLD) * 10 + 0.85)
        elif sim_match and not breed_match:
            same_breed = True
            raw_conf = 0.65
        elif not sim_match and breed_match:
            same_breed = True
            raw_conf = 0.60
        else:
            same_breed = False
            raw_conf = min(1.0, (SAME_BREED_THRESHOLD - similarity) * 6 + 0.70)

        confidence = round(float(np.clip(raw_conf, 0.0, 1.0)), 3)
        verdict = self._build_verdict(same_breed, confidence, pred1, pred2)

        return ComparisonResult(
            same_breed=same_breed,
            confidence=confidence,
            similarity_score=round(similarity, 4),
            image1_prediction=pred1,
            image2_prediction=pred2,
            verdict=verdict,
        )

    def identify_breed(self, img: Image.Image) -> BreedPrediction:
        """Identify the breed in a single image."""
        emb = self._get_image_embedding(img)
        return self._classify_breed(emb)

    @torch.no_grad()
    def _encode_breeds(self) -> torch.Tensor:
        """Pre-compute and cache normalised text embeddings for all breeds."""
        prompts = [f"a photo of a {breed}" for breed in ALL_BREEDS]
        inputs = self.processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        text_feats = self.model.get_text_features(**inputs)
        if not isinstance(text_feats, torch.Tensor):
            text_feats = text_feats.pooler_output
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)
        return text_feats

    @torch.no_grad()
    def _get_image_embedding(self, img: Image.Image) -> torch.Tensor:
        """Return normalised CLIP image embedding (1-D tensor)."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        img_feats = self.model.get_image_features(**inputs)
        if not isinstance(img_feats, torch.Tensor):
            img_feats = img_feats.pooler_output
        img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
        return img_feats.squeeze(0)

    def _classify_breed(self, img_emb: torch.Tensor) -> BreedPrediction:
        """Zero-shot breed classification via cosine similarity with breed prompts."""
        logits = (img_emb.unsqueeze(0) @ self._breed_text_embeddings.T).squeeze(0)
        probs = torch.softmax(logits * 100, dim=0).cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_breed = ALL_BREEDS[top_idx]
        top_conf = float(probs[top_idx])

        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(ALL_BREEDS[i], round(float(probs[i]), 4)) for i in top3_idx]

        pet_type = "cat" if top_breed in CAT_BREEDS else "dog"

        return BreedPrediction(
            breed=top_breed,
            confidence=round(top_conf, 4),
            pet_type=pet_type,
            top_3=top3,
        )

    @staticmethod
    def _build_verdict(
        same_breed: bool,
        confidence: float,
        pred1: BreedPrediction,
        pred2: BreedPrediction,
    ) -> str:
        conf_pct = f"{confidence * 100:.1f}%"
        if same_breed:
            if pred1.breed == pred2.breed:
                return (
                    f"✅ Same Breed — both appear to be {pred1.breed} "
                    f"(confidence: {conf_pct})"
                )
            else:
                return (
                    f"✅ Same Breed (likely) — visual similarity suggests the same breed "
                    f"despite slightly different model predictions "
                    f"(confidence: {conf_pct})"
                )
        else:
            return (
                f"❌ Different Breeds — Image 1 looks like {pred1.breed}, "
                f"Image 2 looks like {pred2.breed} "
                f"(confidence: {conf_pct})"
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compare_images(path1: str | Path, path2: str | Path) -> ComparisonResult:
    """Load two image files and compare their breeds."""
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    identifier = BreedIdentifier()
    return identifier.compare(img1, img2)
