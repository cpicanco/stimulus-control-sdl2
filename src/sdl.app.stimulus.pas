{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , fgl
  , SDL2
  , sdl.app.graphics.rectangule
  , sdl.app.stimulus.contract
  //, sdl.app.choiceable.contract
  , sdl.app.renderer.custom
  , sdl.app.events.abstract
  , sdl.app.stimulus.types
  ;

type

  TChoices = specialize TFPGList<TObject>;

  { TStimulus }

  TStimulus = class(TComponent, IStimulus)
    private
      FPosition: Integer;
      FResponseID : Integer;
      FStimulusID : ShortInt;
      FIndex : integer;
      FIsSample: Boolean;
      FOnMouseDown: TOnMouseEvent;
      FOnMouseEnter: TNotifyEvent;
      FOnMouseExit: TNotifyEvent;
      FOnMouseMove: TOnMouseEvent;
      FOnMouseUp: TOnMouseEvent;
      FOnResponse: TNotifyEvent;
      function GetID : TStimulusID;
      procedure SetIsSample(AValue: Boolean);
      procedure SetOnMouseDown(AValue: TOnMouseEvent);
      procedure SetOnMouseEnter(AValue: TNotifyEvent);
      procedure SetOnMouseExit(AValue: TNotifyEvent);
      procedure SetOnMouseMove(AValue: TOnMouseEvent);
      procedure SetOnMouseUp(AValue: TOnMouseEvent);
      procedure SetOnResponse(AValue: TNotifyEvent);
    protected
      FWord : string;
      function ToData: string;
      function GetRect: TRectangule; virtual; abstract;
      function GetStimulusName : string; virtual; abstract;
      procedure SetRect(AValue: TRectangule); virtual; abstract;
      procedure MouseDown(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer); virtual; abstract;
      procedure MouseUp(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer); virtual; abstract;
      procedure MouseEnter(Sender:TObject); virtual; abstract;
      procedure MouseExit(Sender:TObject); virtual; abstract;
    public
      constructor Create(AOwner : TComponent); override;
      destructor Destroy; override;
      function AsInterface : IStimulus;
      function IsCorrectResponse : Boolean; virtual; abstract;
      procedure Load(AParameters : TStringList;
        AParent : TObject; ARect: TSDL_Rect); virtual; abstract;
      procedure DoResponse; virtual;
      procedure Start; virtual; abstract;
      procedure Stop; virtual; abstract;
      property OnMouseMove: TOnMouseEvent read FOnMouseMove write SetOnMouseMove;
      property OnMouseDown: TOnMouseEvent read FOnMouseDown write SetOnMouseDown;
      property OnMouseUp: TOnMouseEvent read FOnMouseUp write SetOnMouseUp;
      property OnMouseEnter: TNotifyEvent read FOnMouseEnter write SetOnMouseEnter;
      property OnMouseExit: TNotifyEvent read FOnMouseExit write SetOnMouseExit;
      property OnResponse : TNotifyEvent read FOnResponse write SetOnResponse;
      property IsSample : Boolean read FIsSample write SetIsSample;
      property Index : Integer read FIndex write FIndex;
      property Position : Integer read FPosition write FPosition;
      property Rectangule  : TRectangule read GetRect write SetRect;
  end;


implementation

uses session.pool, session.loggers.writerow.timestamp;

{ TStimulus }

procedure TStimulus.SetIsSample(AValue: Boolean);
begin
  if FIsSample=AValue then Exit;
  FIsSample:=AValue;
end;

function TStimulus.GetID: TStimulusID;
begin
  Result.SubjcID := Pool.Counters.Subject;
  Result.SessiID := Pool.Session.ID;
  Result.BlockID := Pool.Session.Block.UID;
  Result.TrialID := Pool.Session.Trial.UID;
  if IsSample then begin
    Result.StimuID := -FIndex;
  end else begin
    Result.StimuID := FIndex;
  end;
  Result.RespoID := FResponseID;
  Result.Name := GetStimulusName;
end;

procedure TStimulus.SetOnMouseDown(AValue: TOnMouseEvent);
begin
  if FOnMouseDown=AValue then Exit;
  FOnMouseDown:=AValue;
end;

procedure TStimulus.SetOnMouseEnter(AValue: TNotifyEvent);
begin
  if FOnMouseEnter=AValue then Exit;
  FOnMouseEnter:=AValue;
end;

procedure TStimulus.SetOnMouseExit(AValue: TNotifyEvent);
begin
  if FOnMouseExit=AValue then Exit;
  FOnMouseExit:=AValue;
end;

procedure TStimulus.SetOnMouseMove(AValue: TOnMouseEvent);
begin
  if FOnMouseMove=AValue then Exit;
  FOnMouseMove:=AValue;
end;

procedure TStimulus.SetOnMouseUp(AValue: TOnMouseEvent);
begin
  if FOnMouseUp=AValue then Exit;
  FOnMouseUp:=AValue;
end;

procedure TStimulus.SetOnResponse(AValue: TNotifyEvent);
begin
  if FOnResponse=AValue then Exit;
  FOnResponse:=AValue;
end;

function TStimulus.ToData: string;
begin
  Result := FWord+'-'+Position.ToString;
end;

constructor TStimulus.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FResponseID := 0;
end;

destructor TStimulus.Destroy;
begin
  inherited Destroy;
end;

function TStimulus.AsInterface: IStimulus;
begin
  Result := Self as IStimulus;
end;

procedure TStimulus.DoResponse;
begin
  Inc(FResponseID);
  Timestamp('Stimulus.Response'+#32+GetID.ToString);
  if Assigned(OnResponse) then
    OnResponse(Self);
end;

end.

