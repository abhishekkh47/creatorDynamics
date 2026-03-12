from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from db_models import Account, Post
from schemas import AccountCreate, AccountResponse
from serializers import feature_dict
from utils import fmt

router = APIRouter(prefix="/accounts", tags=["Accounts"])


@router.post(
    "",
    response_model=AccountResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a creator account",
)
def create_account(req: AccountCreate, db: Session = Depends(get_db)) -> AccountResponse:
    existing = db.query(Account).filter(Account.username == req.username).first()
    if existing:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Account '{req.username}' already exists (id={existing.id}).",
        )
    account = Account(
        username=req.username,
        instagram_id=req.instagram_id,
        follower_count=req.follower_count,
        cluster_tier=req.cluster_tier,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return AccountResponse(
        id=account.id,
        username=account.username,
        instagram_id=account.instagram_id,
        follower_count=account.follower_count,
        cluster_tier=account.cluster_tier,
        created_at=fmt(account.created_at),
        features=None,
        post_count=0,
    )


@router.get(
    "/{account_id}",
    response_model=AccountResponse,
    summary="Get account + current rolling features",
)
def get_account(account_id: int, db: Session = Depends(get_db)) -> AccountResponse:
    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Account {account_id} not found.")
    post_count = db.query(Post).filter(Post.account_id == account_id).count()
    return AccountResponse(
        id=account.id,
        username=account.username,
        instagram_id=account.instagram_id,
        follower_count=account.follower_count,
        cluster_tier=account.cluster_tier,
        created_at=fmt(account.created_at),
        features=feature_dict(account.feature),
        post_count=post_count,
    )
