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
  , sdl.app.stimuli.contract;

type

  { TStimuli }

  TStimuli = class(TComponent, IStimuli)
  private
    FOnFinalize: TNotifyEvent;
    FSchedule : TSchedule;
    FOnConsequence: TNotifyEvent;
    FOnResponse: TNotifyEvent;
    FOnStop: TNotifyEvent;
    procedure Consequence(Sender : TObject);
    procedure Response(Sender : TObject);
    procedure SetOnConsequence(AValue: TNotifyEvent);
    procedure SetOnFinalize(AValue: TNotifyEvent);
    procedure SetOnResponse(AValue: TNotifyEvent);
    procedure SetOnStop(AValue : TNotifyEvent);
  protected
    function CustomName : string;
    //function ContainerItems : IEnumerable; virtual; abstract;
    procedure DoExpectedResponse; virtual; abstract;
    procedure Load(AParameters: TStringList; AParent: TObject); virtual;
    procedure Start; virtual; abstract;
    procedure Stop; virtual; abstract;
    procedure SetSchedule(AValue: TSchedule); virtual;
  public
    constructor Create(AOwner : TComponent); overload; override;
    constructor Create(AOwner : TComponent;
      ASchedule : TSchedule); virtual; overload;
    function AsString : string; virtual;
    function AsInterface: IStimuli;
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
  Result := Self.ClassName.Replace('T', '');
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

procedure TStimuli.Load(AParameters: TStringList; AParent: TObject);
begin
  if not Assigned(AParent) then
    raise Exception.Create('You must assigned a parent before loading.');
end;

procedure TStimuli.SetSchedule(AValue: TSchedule);
begin
  if FSchedule=AValue then Exit;
  FSchedule:=AValue;
end;

constructor TStimuli.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FSchedule := TSchedule.Create(Self);
  with FSchedule do begin
    OnConsequence:= @Self.Consequence;
    OnResponse:= @Self.Response;
  end;
end;

constructor TStimuli.Create(AOwner : TComponent; ASchedule : TSchedule);
begin
  inherited Create(AOwner);
  FSchedule := ASchedule;
end;

function TStimuli.AsString: string;
begin
  Result := ClassName;
end;

function TStimuli.AsInterface: IStimuli;
begin
  Result := Self as IStimuli;
end;


end.

