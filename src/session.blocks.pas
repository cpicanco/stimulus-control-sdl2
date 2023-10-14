{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.blocks;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

type

  { TBlock }

  {
    Controls Trials Create/Destroy cycle
  }
  TBlock = class(TComponent)
  private
    FOnEndBlock: TNotifyEvent;
    FOnInterTrialEnd: TNotifyEvent;
    procedure SetOnEndBlock(AValue: TNotifyEvent);
    procedure InterTrialEventsEnd(Sender: TObject);
    procedure SetOnInterTrialEnd(AValue: TNotifyEvent);
    procedure EndBlock;
  public
    constructor Create(AOwner : TComponent); override;
    procedure BeforePlay;
    procedure Play;
    property OnEndBlock : TNotifyEvent read FOnEndBlock write SetOnEndBlock;
    property OnInterTrialEnd : TNotifyEvent read FOnInterTrialEnd write SetOnInterTrialEnd;
  end;

implementation

uses sdl.app.trials.factory
   , session.intertrial
   , session.pool
   ;

{ TBlock }

procedure TBlock.SetOnEndBlock(AValue: TNotifyEvent);
begin
  if FOnEndBlock=AValue then Exit;
  FOnEndBlock:=AValue;
end;

procedure TBlock.InterTrialEventsEnd(Sender: TObject);
begin
  Pool.EndCriteria.OfTrial;
  Play;
end;

procedure TBlock.SetOnInterTrialEnd(AValue: TNotifyEvent);
begin
  if FOnInterTrialEnd=AValue then Exit;
  FOnInterTrialEnd:=AValue;
end;

procedure TBlock.EndBlock;
begin
  if Assigned(OnEndBlock) then
    OnEndBlock(Self);
end;

constructor TBlock.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  InterTrial := TInterTrialEvents.Create(Self);
  InterTrial.OnEnd:=@InterTrialEventsEnd;
end;

procedure TBlock.BeforePlay;
begin
  Pool.EndCriteria.InvalidateBlock;
end;

procedure TBlock.Play;
begin
  if Pool.EndCriteria.OfBlock then begin
    EndBlock;
  end else begin
    TTrialFactory.Play;
  end;
end;

end.
