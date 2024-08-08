{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimuli;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Controls, Schedules
  , sdl.app.stimuli.contract
  , sdl.app.trials.types
  , sdl.app.navigable.contract
  , sdl.app.selectable.list;

type

  { TStimuli }

  TStimuli = class(TObject, IStimuli)
  private
    FOnFinalize: TNotifyEvent;
    FSchedule : TSchedule;
    FOnConsequence: TNotifyEvent;
    FOnResponse: TNotifyEvent;
    FOnStop: TNotifyEvent;
    FTrial: TObject;
    function Selectables : TSelectables;
    procedure Consequence(Sender : TObject);
    procedure Response(Sender : TObject);
    procedure SetOnConsequence(AValue: TNotifyEvent);
    procedure SetOnFinalize(AValue: TNotifyEvent);
    procedure SetOnResponse(AValue: TNotifyEvent);
    procedure SetOnStop(AValue : TNotifyEvent);
    procedure SetTrial(AValue: TObject);
  protected
    FSelectables : TSelectables;
    FResponse : string;
    function GetTrial : TObject;
    function CustomName : string; virtual;
    function MyResult : TTrialResult; virtual;
    function Header : string; virtual;
    function ToData : string; virtual;
    //function ContainerItems : IEnumerable; virtual; abstract;
    procedure DoExpectedResponse; virtual; abstract;
    procedure Load(AParameters: TStringList; AParent: TObject); virtual;
    procedure Start; virtual; abstract;
    procedure Stop; virtual; abstract;
    procedure SetSchedule(AValue: TSchedule); virtual;
  public
    constructor Create; virtual; overload;
    constructor Create(ASchedule : TSchedule); virtual; overload;
    destructor Destroy; override;
    function AsString : string; virtual;
    function AsIStimuli: IStimuli;
    function AsINavigable: INavigable; virtual;
    function IsStarter : Boolean; virtual;
    property Trial : TObject read GetTrial write SetTrial;
    property Schedule : TSchedule read FSchedule write SetSchedule;
    property OnStop : TNotifyEvent read FOnStop write SetOnStop;
    property OnConsequence : TNotifyEvent read FOnConsequence write SetOnConsequence;
    property OnResponse : TNotifyEvent read FOnResponse write SetOnResponse;
    property OnFinalize : TNotifyEvent read FOnFinalize write SetOnFinalize;
  end;

implementation

{ TStimuli }

function TStimuli.CustomName: string;
begin
  Result := Self.ClassName.Replace('T', '', []);
end;

function TStimuli.MyResult: TTrialResult;
begin
  Result := None;
end;

function TStimuli.Header: string;
begin
  Result := 'Response';
end;

function TStimuli.ToData: string;
begin
  Result := FResponse;
end;

function TStimuli.GetTrial: TObject;
begin
  Result := FTrial;
end;

function TStimuli.Selectables: TSelectables;
begin
  Result := FSelectables;
end;

procedure TStimuli.Consequence(Sender: TObject);
begin
  if Assigned(OnConsequence) then OnConsequence(Self);
end;

procedure TStimuli.Response(Sender: TObject);
begin
  if Assigned(OnResponse) then OnResponse(Self);
end;

procedure TStimuli.SetOnConsequence(AValue: TNotifyEvent);
begin
  if FOnConsequence=AValue then Exit;
  FOnConsequence:=AValue;
end;

procedure TStimuli.SetOnFinalize(AValue: TNotifyEvent);
begin
  if FOnFinalize=AValue then Exit;
  FOnFinalize:=AValue;
end;

procedure TStimuli.SetOnResponse(AValue: TNotifyEvent);
begin
  if FOnResponse=AValue then Exit;
  FOnResponse:=AValue;
end;

procedure TStimuli.SetOnStop(AValue: TNotifyEvent);
begin
  if FOnStop=AValue then Exit;
  FOnStop:=AValue;
end;

procedure TStimuli.SetTrial(AValue: TObject);
begin
  if FTrial = AValue then Exit;
  FTrial := AValue;
end;

procedure TStimuli.Load(AParameters: TStringList; AParent: TObject);
begin
  if not Assigned(AParent) then
    raise Exception.Create('You must assigned a parent before loading.');
end;

procedure TStimuli.SetSchedule(AValue: TSchedule);
begin
  if FSchedule.AsString=AValue.AsString then Exit;
  FSchedule.Load(AValue.AsString);
  with FSchedule do begin
    OnConsequence:= AValue.OnConsequence;
    OnResponse:= AValue.OnResponse;
  end;
end;

constructor TStimuli.Create;
begin
  inherited Create;
  FSelectables := TSelectables.Create;
  FResponse := '';
  FSchedule := TSchedule.Create(nil);
  with FSchedule do begin
    OnConsequence:= @Self.Consequence;
    OnResponse:= @Self.Response;
  end;
end;

constructor TStimuli.Create(ASchedule : TSchedule);
begin
  Create;
  FSchedule.Load(ASchedule.AsString);
  with FSchedule do begin
    OnConsequence:= ASchedule.OnConsequence;
    OnResponse:= ASchedule.OnResponse;
  end;
end;

destructor TStimuli.Destroy;
begin
  FSelectables.Free;
  FSchedule.Free;
  FOnFinalize := nil;
  FOnConsequence := nil;
  FOnResponse := nil;
  FOnStop := nil;
  inherited Destroy;
end;

function TStimuli.AsString: string;
begin
  Result := ClassName;
end;

function TStimuli.AsIStimuli: IStimuli;
begin
  Result := Self as IStimuli;
end;

function TStimuli.AsINavigable: INavigable;
begin
  Result := nil;
end;

function TStimuli.IsStarter: Boolean;
begin
  Result := False;
end;


end.

