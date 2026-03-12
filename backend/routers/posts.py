from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from db_models import Account, Post, Prediction
from feature_engine import upsert_feature_store
from predictor import model_store, predict_stage1, predict_stage2
from schemas import PostIngest, PostResponse, ReachUpdate, ReachUpdateResponse, VelocityUpdate
from serializers import prediction_to_summary
from utils import fmt, utcnow

router = APIRouter(tags=["Posts"])


@router.post(
    "/accounts/{account_id}/posts",
    response_model=PostResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a new post → auto Stage-1 prediction",
    description=(
        "Record a new post going live. If the account's feature store has been "
        "populated (≥2 historical posts with known reach), Stage-1 prediction "
        "fires automatically and is returned embedded in the response.\n\n"
        "Call `PATCH /posts/{id}/velocity` at T+1h and "
        "`PATCH /posts/{id}/reach` at T+24h to complete the lifecycle."
    ),
)
def ingest_post(
    account_id: int,
    req: PostIngest,
    db: Session = Depends(get_db),
) -> PostResponse:
    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Account {account_id} not found.")

    post = Post(
        account_id=account_id,
        instagram_post_id=req.instagram_post_id,
        posted_at=req.posted_at,
        content_quality=req.content_quality,
        cluster_id=req.cluster_id,
    )
    db.add(post)
    db.commit()
    db.refresh(post)

    # Auto Stage-1 prediction when feature store is populated
    pred_row = None
    if model_store.stage1 is not None and account.feature is not None:
        f = account.feature
        s1 = predict_stage1(
            rolling_weighted_median=f.rolling_weighted_median,
            rolling_volatility=f.rolling_volatility,
            posting_frequency=f.posting_frequency,
            cluster_entropy=f.cluster_entropy,
            content_quality=req.content_quality,
            cluster_id=req.cluster_id,
            hour_of_day=req.posted_at.hour,
        )
        pred_row = Prediction(
            account_id=account_id,
            post_id=post.id,
            stage1_prob=s1["survival_probability"],
            stage1_survives=s1["survives"],
            stage1_confidence=s1["confidence"],
            stage1_called_at=utcnow(),
            feat_rolling_weighted_median=f.rolling_weighted_median,
            feat_rolling_volatility=f.rolling_volatility,
            feat_posting_frequency=f.posting_frequency,
            feat_cluster_entropy=f.cluster_entropy,
            feat_content_quality=req.content_quality,
            feat_cluster_id=req.cluster_id,
            feat_posting_time_bucket=s1["posting_time_bucket"],
        )
        db.add(pred_row)
        db.commit()
        db.refresh(pred_row)

    return PostResponse(
        id=post.id,
        account_id=post.account_id,
        instagram_post_id=post.instagram_post_id,
        posted_at=fmt(post.posted_at),
        content_quality=post.content_quality,
        cluster_id=post.cluster_id,
        reach_24h=None,
        likes_1h=None,
        comments_1h=None,
        created_at=fmt(post.created_at),
        prediction=prediction_to_summary(pred_row),
    )


@router.get(
    "/posts/{post_id}",
    response_model=PostResponse,
    summary="Get post details + linked prediction",
)
def get_post(post_id: int, db: Session = Depends(get_db)) -> PostResponse:
    post = db.get(Post, post_id)
    if post is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Post {post_id} not found.")
    return PostResponse(
        id=post.id,
        account_id=post.account_id,
        instagram_post_id=post.instagram_post_id,
        posted_at=fmt(post.posted_at),
        content_quality=post.content_quality,
        cluster_id=post.cluster_id,
        reach_24h=post.reach_24h,
        likes_1h=post.likes_1h,
        comments_1h=post.comments_1h,
        created_at=fmt(post.created_at),
        prediction=prediction_to_summary(post.prediction),
    )


@router.patch(
    "/posts/{post_id}/velocity",
    response_model=PostResponse,
    summary="Record 1h engagement → auto Stage-2 prediction",
    description=(
        "Call this ~60 minutes after posting with raw like and comment counts. "
        "Stage-2 prediction fires automatically and updates the linked Prediction row."
    ),
)
def update_velocity(
    post_id: int,
    req: VelocityUpdate,
    db: Session = Depends(get_db),
) -> PostResponse:
    post = db.get(Post, post_id)
    if post is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Post {post_id} not found.")

    post.likes_1h    = req.likes_1h
    post.comments_1h = req.comments_1h
    db.commit()

    pred = post.prediction
    if pred is None or pred.stage1_prob is None:
        db.refresh(post)
        return PostResponse(
            id=post.id,
            account_id=post.account_id,
            instagram_post_id=post.instagram_post_id,
            posted_at=fmt(post.posted_at),
            content_quality=post.content_quality,
            cluster_id=post.cluster_id,
            reach_24h=post.reach_24h,
            likes_1h=post.likes_1h,
            comments_1h=post.comments_1h,
            created_at=fmt(post.created_at),
            prediction=prediction_to_summary(pred),
        )

    account = db.get(Account, post.account_id)
    # Use the baseline that was snapshotted at prediction time for consistency
    baseline = pred.feat_rolling_weighted_median or (
        account.feature.rolling_weighted_median if account.feature else 8000.0
    )

    s2 = predict_stage2(
        stage1_prior=pred.stage1_prob,
        rolling_weighted_median=baseline,
        likes_1h=req.likes_1h,
        comments_1h=req.comments_1h,
        cluster_tier=account.cluster_tier,
    )

    pred.stage2_prob       = s2["survival_probability"]
    pred.stage2_survives   = s2["survives"]
    pred.stage2_correction = s2["correction"]
    pred.stage2_confidence = s2["confidence"]
    pred.stage2_called_at  = utcnow()
    vel = s2["velocity_features"]
    pred.vel_norm_likes_1h    = vel["norm_likes_1h"]
    pred.vel_comment_ratio_1h = vel["comment_ratio_1h"]
    pred.vel_on_track_score   = vel["on_track_score"]
    db.commit()
    db.refresh(post)

    return PostResponse(
        id=post.id,
        account_id=post.account_id,
        instagram_post_id=post.instagram_post_id,
        posted_at=fmt(post.posted_at),
        content_quality=post.content_quality,
        cluster_id=post.cluster_id,
        reach_24h=post.reach_24h,
        likes_1h=post.likes_1h,
        comments_1h=post.comments_1h,
        created_at=fmt(post.created_at),
        prediction=prediction_to_summary(pred),
    )


@router.patch(
    "/posts/{post_id}/reach",
    response_model=ReachUpdateResponse,
    summary="Record 24h reach → close lifecycle + recompute feature store",
    description=(
        "Record the final 24h reach. This:\n"
        "1. Marks the prediction as correct/incorrect\n"
        "2. Recomputes the feature store so the account's next prediction uses updated baselines"
    ),
)
def update_reach(
    post_id: int,
    req: ReachUpdate,
    db: Session = Depends(get_db),
) -> ReachUpdateResponse:
    post = db.get(Post, post_id)
    if post is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Post {post_id} not found.")

    post.reach_24h    = req.reach_24h
    post.likes_24h    = req.likes_24h
    post.comments_24h = req.comments_24h
    db.commit()

    pred = post.prediction
    baseline_at_time = pred.feat_rolling_weighted_median if pred else None
    actual_survived  = False

    if pred is not None and baseline_at_time is not None:
        actual_survived          = req.reach_24h > baseline_at_time
        pred.actual_survived     = actual_survived
        pred.outcome_recorded_at = utcnow()
        db.commit()

    fs = upsert_feature_store(post.account_id, db)

    s1_correct = s2_correct = None
    if pred is not None and pred.actual_survived is not None:
        if pred.stage1_survives is not None:
            s1_correct = pred.stage1_survives == pred.actual_survived
        if pred.stage2_survives is not None:
            s2_correct = pred.stage2_survives == pred.actual_survived

    return ReachUpdateResponse(
        post_id=post.id,
        reach_24h=req.reach_24h,
        actual_survived=actual_survived,
        rolling_weighted_median_at_time=baseline_at_time,
        stage1_correct=s1_correct,
        stage2_correct=s2_correct,
        feature_store_updated=fs is not None,
    )
